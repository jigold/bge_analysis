import asyncio
import os
import io
import base64
import json
import sys
import pandas as pd
from collections import defaultdict, namedtuple
from jinja2 import Environment, StrictUndefined
from typing import Dict, Optional, Tuple
from functools import partial

import hailtop.batch as hb
import hailtop.fs as hfs
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import bounded_gather

from ..globals import file_exists
from .jobs import flare, mt_to_vcf, vcf_to_mt, union_sample_groups_from_vcfs
from .sample_group import FlareSampleGroup, split_samples_into_groups


env = Environment(undefined=StrictUndefined)


ContigData = namedtuple('ContigData', ['contig', 'input_file', 'reference_file', 'map_file'])


async def run_sample_group(b: hb.Batch,
                           args: dict,
                           contig_data: Dict[str, ContigData],
                           sample_group: FlareSampleGroup,
                           model_file: Optional[hb.ResourceFile],
                           reference_panel: hb.ResourceFile,
                           fs: RouterAsyncFS) -> Tuple[Dict[str, hb.Job], str]:
    print(f'staging sample group {sample_group.name}')

    skip_flare = False

    flare_already_completed = [False for _ in contig_data.keys()]

    flare_output_files = sample_group.get_flare_mts(list(contig_data.keys()))

    if args['use_checkpoints']:
        flare_already_completed = await bounded_gather(*[partial(file_exists, fs, output_file_root + '/_SUCCESS')
                                                                for output_file_root in flare_output_files],
                                                                cancel_on_error=True)

        skip_flare = all(flare_already_completed)

    vcf_to_mt_jobs = {}

    if not skip_flare:
        samples_list = sample_group.write_samples_list()
        samples_list_input = b.read_input(samples_list)

        for idx, (contig, data) in enumerate(contig_data.items()):
            if args['use_checkpoints'] and flare_already_completed[idx]:
                continue

            if data.input_file.endswith('.mt'):
                mt_to_vcf_j = mt_to_vcf(b,
                                        sample_group,
                                        data.input_file,
                                        contig,
                                        args['docker_hail'],
                                        args['mt_to_vcf_cpu'],
                                        args['mt_to_vcf_memory'],
                                        args['mt_to_vcf_storage'],
                                        )
                input_vcf_file = mt_to_vcf_j.vcf_path
            else:
                input_vcf_file = b.read_input(data.input_file)

            flare_output_root = sample_group.flare_output_file_root(contig)
            flare_exists = flare_already_completed[idx]

            map_file_input = b.read_input(data.map_file)
            reference_input = b.read_input(data.reference_file)

            flare_j = flare(b,
                            flare_output_root,
                            flare_exists,
                            contig,
                            sample_group,
                            input_vcf_file,
                            samples_list_input,
                            reference_input,
                            reference_panel,
                            map_file_input,
                            model_file,
                            args['docker_flare'],
                            args['flare_cpu'],
                            args['flare_memory'],
                            args['flare_storage'],
                            args['use_checkpoints'])

            if model_file is None:
                model_file = flare_j.flare.model

            vcf_to_mt_j = vcf_to_mt(b,
                                    sample_group,
                                    flare_j.flare.vcf,
                                    sample_group.flare_mt(contig),
                                    contig,
                                    args['docker_hail'],
                                    args['vcf_to_mt_cpu'],
                                    args['vcf_to_mt_memory'],
                                    args['vcf_to_mt_storage'],
                                    args['use_checkpoints']
                                    )

            vcf_to_mt_jobs[contig] = vcf_to_mt_j

    return (vcf_to_mt_jobs, model_file)


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

    with hfs.open(args['manifest_file'], 'r') as f:
        manifest = pd.read_csv(io.StringIO(f.read()), sep='\t')

    contigs = manifest[args['contig_col']].to_list()
    input_files = manifest[args['input_file_col']].to_list()
    reference_files = manifest[args['reference_file_col']].to_list()
    map_files = manifest[args['map_file_col']].to_list()

    reference_panel = b.read_input(args['reference_panel'])

    input_data_by_contig = {}
    for contig, input_file, reference_file, map_file in zip(contigs, input_files, reference_files, map_files):
        input_data_by_contig[contig] = ContigData(contig,
                                        input_file,
                                        b.read_input(reference_file),
                                        b.read_input(map_file))

    sample_groups = split_samples_into_groups(args['sample_list'],
                                              args['sample_group_size'],
                                              args['staging_remote_tmpdir'])
    if args['sample_group_index'] is not None:
        sample_groups = [sg for sg in sample_groups if sg.sample_group_index == args['sample_group_index']]
        assert sample_groups

    print(f'found {len(sample_groups)} sample groups')

    if args['model'] is not None:
        model_file = b.read_input(args['model'])
    else:
        model_file = None

    union_ligate_input_jobs = defaultdict(list)
    for sample_group in sample_groups:
        flare_mt_jobs, model_file = await run_sample_group(b,
                                                           args,
                                                           input_data_by_contig,
                                                           sample_group,
                                                           model_file,
                                                           reference_panel,
                                                           backend._fs)

        for contig, flare_mt_j in flare_mt_jobs.items():
            union_ligate_input_jobs[contig].append(flare_mt_j)

    for contig in input_data_by_contig.keys():
        sample_group_mts = [sample_group.flare_mt(contig) for sample_group in sample_groups]

        union_sample_groups_inputs_path = args['staging_remote_tmpdir'].rstrip('/') + f'/{contig}/sample_group_mts.txt'
        with hfs.open(union_sample_groups_inputs_path, 'w') as f:
            for mt in sample_group_mts:
                f.write(f'{mt}\n')

        output_file = env.from_string(args['output_file']).render(contig=contig)

        union_j = union_sample_groups_from_vcfs(b,
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
                                                contig,
                                                args['n_partitions'])

        if union_j is not None:
            union_j.depends_on(*union_ligate_input_jobs.get(contig, []))

    b.run(wait=False, disable_progress_bar=True)

    backend.close()


if __name__ == '__main__':
    arguments_b64_str = base64.b64decode(sys.argv[1])
    args = json.loads(arguments_b64_str)

    print(json.dumps(args, indent=4))
    asyncio.run(local_ancestry(args))