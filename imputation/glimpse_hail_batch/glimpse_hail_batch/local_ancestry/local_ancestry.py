import os
import re
from collections import defaultdict
from jinja2 import Environment, StrictUndefined
from typing import Dict, List, Optional
from functools import partial

import hailtop.batch as hb
import hailtop.fs as hfs
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import bounded_gather

from ..globals import Chunk, find_chunks, file_exists

# Need a FLARE docker image

# Read contig chunks

# Need a way to take an input matrix table, export to VCF or take a VCF directly

# Need a path to the reference panel for each chunk

# Pops file


env = Environment(undefined=StrictUndefined)


class FlareSampleGroup:
    def __init__(self, samples: List[str], sample_group_index: int, output_dir: str):
        self.samples = samples
        self.sample_group_index = sample_group_index
        self._output_dir = output_dir

    @property
    def temp_dir(self):
        return f'{self._output_dir}/sg-{self.sample_group_index}'

    @property
    def samples_list_file(self):
        return f'{self.temp_dir}/flare/samples.txt'

    @property
    def name(self):
        return f'sample-group-{self.sample_group_index}'

    def flare_output_file_root(self, contig: str, chunk_index: int):
        return f'{self.temp_dir}/flare/{contig}/chunk-{chunk_index}'

    def write_samples_list(self) -> str:
        with hfs.open(self.samples_list_file, 'w') as f:
            for sample_id in self.samples:
                f.write(f'{sample_id}\n')
        return self.samples_list_file

    def get_flare_output_file_names(self, contig_chunks: Dict[str, List[Chunk]]) -> Dict[str, List[str]]:
        output_files = defaultdict(list)
        for contig, chunks in contig_chunks.items():
            for chunk in chunks:
                output_files[contig].append(self.flare_output_file_root(contig, chunk.chunk_idx))
        return output_files


def flare(b: hb.Batch,
          output_dir: str,
          flare_file_exists: bool,
          sample_group: FlareSampleGroup,
          phased_input: hb.ResourceFile,
          chunk: Chunk,
          samples_list: hb.ResourceFile,
          reference_panel: hb.ResourceFile,
          map_file: hb.ResourceFile,
          model: Optional[hb.ResourceFile],
          docker: str,
          cpu: int,
          memory: str,
          use_checkpoint: bool) -> Optional[hb.Job]:
    sample_group_index = sample_group.sample_group_index

    if use_checkpoint and flare_file_exists:
        return None

    j = b.new_bash_job(name=f'flare/sample-group-{sample_group_index}/{chunk.chunk_contig}/{chunk.chunk_idx}',
                       attributes={'sample-group-index': str(sample_group_index),
                                   'contig': str(chunk.chunk_contig),
                                   'chunk-index': str(chunk.chunk_idx),
                                   'task': 'flare'})

    j.image(docker)
    j.storage('20Gi')
    j.cpu(cpu)
    j.memory(memory)

    if memory == 'lowmem':
        memory_gib = int(0.75 * cpu) + 1
    elif memory == 'standard':
        memory_gib = int(3.5 * cpu) + 1
    else:
        assert memory == 'highmem', memory
        memory_gib = int(7.5 * cpu) + 1

    reference = b.read_input(chunk.path)

    j.declare_resource_group(flare={'vcf': '{root}.anc.vcf.gz',
                                    'log': '{root}.log',
                                    'model': '{root}.model',
                                    'global_ancestry': '{root}.global.anc.gz'})

    if model is not None:
        model_flag = f'model={model}'
    else:
        model_flag = ''

    flare_cmd = f'''
set -e

bcftools convert --threads {cpu} -O z -o {j.reference} {reference}
bcftools convert --threads {cpu} -O z -o {j.gt} {phased_input}

java -Xmx{memory_gib}g -jar flare.jar \
    ref={j.reference} \
    ref-panel={reference_panel} \
    gt={j.gt} \
    map={map_file} \
    out={j.output} \
    probs=true \
    gt-samples={samples_list} \
    nthreads={cpu} \
    seed=14235432 \
    {model_flag}
'''

    j.command(flare_cmd)

    b.write_output(j.flare, output_dir)

    return j


async def run_sample_group(b: hb.Batch,
                           args: dict,
                           contig_chunks: Dict[str, List[Chunk]],
                           sample_group: FlareSampleGroup,
                           fs: RouterAsyncFS) -> Dict[str, List[hb.Job]]:
    print(f'staging sample group {sample_group.name}')

    skip_flare = False

    flare_already_completed = [False for contig, chunks in contig_chunks.items() for _ in chunks]

    flare_output_files = sample_group.get_flare_output_file_names(contig_chunks)

    if args['use_checkpoints']:
        flare_already_completed = await bounded_gather(*[partial(file_exists, fs, flare_output_files[contig][chunk_idx] + '.anc.vcf.gz')
                                                                for contig, chunks in contig_chunks.items()
                                                                for chunk_idx, chunk in enumerate(chunks)],
                                                                cancel_on_error=True)

        skip_flare = all(flare_already_completed)

    flare_jobs = defaultdict(list)

    if not skip_flare:
        samples_list = sample_group.write_samples_list()
        samples_list_input = b.read_input(samples_list)

        global_chunk_idx = 0
        for contig, chunks in contig_chunks.items():
            local_chunk_idx = 0
            for chunk in chunks:
                flare_output_root = flare_output_files[contig][local_chunk_idx]

                flare_exists = flare_already_completed[global_chunk_idx]

                model_file = None

                phased_input = sample_group.phased_input(chunk)

                flare_j = flare(b,
                                flare_output_root,
                                flare_exists,
                                sample_group,
                                phased_input,
                                chunk,
                                samples_list_input,
                                reference_panel_input,
                                map_file,
                                model_file,
                                args['docker_flare'],
                                args['flare_cpu'],
                                args['flare_memory'],
                                args['use_checkpoints'])

                flare_jobs[contig].append(flare_j)

                global_chunk_idx += 1
                local_chunk_idx += 1

    return flare_jobs

async def local_ancestry(args: dict):
    if hfs.exists(args['output_file']):
        raise Exception(f'output file {args["output_file"]} already exists.')

    batch_regions = args['batch_regions']
    if batch_regions is not None:
        batch_regions = batch_regions.split(',')

    batch_name = args['batch_name'] or 'flare'

    backend = hb.ServiceBackend(billing_project=args['billing_project'],
                                remote_tmpdir=args['batch_remote_tmpdir'],
                                regions=batch_regions,
                                gcs_requester_pays_configuration=args['gcs_requester_pays_configuration'])

    batch_id = args['batch_id'] or os.environ.get('HAIL_BATCH_ID')
    if batch_id is not None:
        b = hb.Batch.from_batch_id(int(batch_id),
                                   backend=backend,
                                   requester_pays_project=args['gcs_requester_pays_configuration'])
    else:
        b = hb.Batch(name=batch_name,
                     backend=backend,
                     requester_pays_project=args['gcs_requester_pays_configuration'])

    mount_point = '/crams/'

    # get list of samples from the matrix table
    # divide up into sample groups

    samples = find_crams(args['sample_manifest'],
                         args['sample_id_col'],
                         args['cram_path_col'],
                         args['cram_index_path_col'],
                         args['sex_col'],
                         args['female_code'],
                         args['n_samples'])

    sample_groups = split_samples_into_groups(samples,
                                              args['sample_group_size'],
                                              args['staging_remote_tmpdir'],
                                              mount_point)
    if args['sample_group_index'] is not None:
        sample_groups = [sg for sg in sample_groups if sg.sample_group_index == args['sample_group_index']]
        assert sample_groups

    non_par_contigs = args['non_par_contigs']
    if non_par_contigs is None:
        non_par_contigs = []
    else:
        non_par_contigs = non_par_contigs.split(',')

    chunks = find_chunks(args['reference_dir'],
                         args['chunk_info_dir'],
                         re.compile(args['binary_reference_file_regex']),
                         re.compile(args['chunk_file_regex']),
                         non_par_contigs=non_par_contigs,
                         requested_contig=args['contig'],
                         requested_chunk_index=args['chunk_index'],
                         requester_pays_config=args['gcs_requester_pays_configuration'])

    print(f'found {len(chunks)} chunks')
    print(f'found {len(sample_groups)} sample groups')

    contig_chunks = defaultdict(list)
    for chunk in chunks:
        contig_chunks[chunk.chunk_contig].append(chunk)

    prev_copy_cram_jobs = []
    union_ligate_input_jobs = defaultdict(list)
    for sample_group in sample_groups:
        prev_copy_cram_jobs, ligate_jobs = await run_sample_group(b,
                                                                  args,
                                                                  contig_chunks,
                                                                  sample_group,
                                                                  fasta_input,
                                                                  ref_dict,
                                                                  args['samples_per_copy_group'],
                                                                  prev_copy_cram_jobs,
                                                                  backend._fs)

        for contig, ligate_j in ligate_jobs.items():
            union_ligate_input_jobs[contig].append(ligate_j)

    union_sample_groups_jg = b.create_job_group(attributes={'name': 'union-sample-groups'})

    for contig in contig_chunks.keys():
        union_contig_jg = union_sample_groups_jg.create_job_group(attributes={'name': f'union-sample-groups/{contig}',
                                                                              'contig': contig})

        sample_group_vcfs = [sample_group.ligate_output_file_root(contig) + '.vcf.bgz' for sample_group in sample_groups]
        sample_group_sizes = [sample_group.n_samples for sample_group in sample_groups]

        union_sample_groups_inputs_path = args['staging_remote_tmpdir'].rstrip('/') + f'/{contig}/sample_group_vcfs.txt'
        with hfs.open(union_sample_groups_inputs_path, 'w') as f:
            for vcf, sample_size in zip(sample_group_vcfs, sample_group_sizes):
                f.write(f'{vcf}\t{sample_size}\n')

        output_file = env.from_string(args['output_file']).render(contig=contig)

        union_j = union_sample_groups_from_vcfs(b,
                                                union_contig_jg,
                                                b.read_input(union_sample_groups_inputs_path),
                                                output_file,
                                                args['docker_hail'],
                                                args['merge_vcf_cpu'],
                                                args['merge_vcf_memory'],
                                                args['merge_vcf_storage'],
                                                args['billing_project'],
                                                args['batch_remote_tmpdir'],
                                                batch_regions,
                                                args['use_checkpoints'],
                                                contig)

        if union_j is not None:
            union_j.depends_on(*union_ligate_input_jobs.get(contig, []))

    b.run(wait=False, disable_progress_bar=True)

    backend.close()


if __name__ == '__main__':
    arguments_b64_str = base64.b64decode(sys.argv[1])
    args = json.loads(arguments_b64_str)

    print(json.dumps(args, indent=4))
    asyncio.run(impute(args))