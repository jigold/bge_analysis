from typing import List, Optional

import hailtop.batch as hb
from hailtop.batch.job import Job
import hailtop.fs as hfs

from .sample_group import FlareSampleGroup


def flare(b: hb.Batch,
          output_dir: str,
          flare_file_exists: bool,
          contig: str,
          sample_group: FlareSampleGroup,
          phased_input: hb.ResourceFile,
          samples_list: hb.ResourceFile,
          reference: hb.ResourceFile,
          reference_panel: hb.ResourceFile,
          map_file: hb.ResourceFile,
          model: Optional[hb.ResourceFile],
          docker: str,
          cpu: int,
          memory: str,
          storage: str,
          use_checkpoint: bool) -> Optional[hb.Job]:
    sample_group_index = sample_group.sample_group_index

    if use_checkpoint and flare_file_exists:
        return None

    j = b.new_bash_job(name=f'flare/sample-group-{sample_group_index}/{contig}',
                       attributes={'sample-group-index': str(sample_group_index),
                                   'contig': contig,
                                   'task': 'flare'})

    j.image(docker)
    j.storage(storage)
    j.cpu(cpu)
    j.memory(memory)

    if memory == 'lowmem':
        memory_gib = int(0.75 * cpu) + 1
    elif memory == 'standard':
        memory_gib = int(3.5 * cpu) + 1
    else:
        assert memory == 'highmem', memory
        memory_gib = int(7.5 * cpu) + 1

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

java -Xmx{memory_gib}g -jar flare.jar \
    ref={reference} \
    ref-panel={reference_panel} \
    gt={phased_input} \
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


def _mt_to_vcf(input_mt: str, output_vcf_path: str, cpu: int):
    import hail as hl

    hl.init(backend="spark",
            local=f"local[{cpu}]",
            default_reference="GRCh38",
            tmp_dir="/io/",
            local_tmpdir="/io/",
            spark_conf={"spark.executor.memory": "7g", "spark.driver.memory": "7g", "spark.driver.maxResultSize": "7g"}
            )

    assert output_vcf_path.endswith('.vcf.bgz')

    mt = hl.read_matrix_table(input_mt)

    hl.export_vcf(mt, output_vcf_path, tabix=True)


def mt_to_vcf(b: hb.Batch,
              sample_group: FlareSampleGroup,
              input_mt: str,
              contig: str,
              docker: str,
              cpu: int,
              memory: str,
              storage: str) -> Optional[Job]:
    j = b.new_python_job(attributes={'name': f'mt-to-vcf/sample-group-{sample_group.sample_group_index}/{contig}',
                                     'task': 'mt-to-vcf'})
    j.cpu(cpu)
    j.image(docker)
    j.storage(storage)
    j.memory(memory)

    j.call(_mt_to_vcf, input_mt, j.vcf_path, cpu)

    return j


def _vcf_to_mt(input_vcf: str, output_path: str, cpu: int):
    import hail as hl

    hl.init(backend="spark",
            local=f"local[{cpu}]",
            default_reference="GRCh38",
            tmp_dir="/io/",
            local_tmpdir="/io/",
            spark_conf={"spark.executor.memory": "7g", "spark.driver.memory": "7g", "spark.driver.maxResultSize": "7g"}
            )

    mt = hl.import_vcf(input_vcf)
    mt.write(output_path, overwrite=True)


def vcf_to_mt(b: hb.Batch,
              sample_group: FlareSampleGroup,
              input_vcf: str,
              output_path: str,
              contig: str,
              docker: str,
              cpu: int,
              memory: str,
              storage: str,
              use_checkpoint: bool) -> Optional[Job]:
    if use_checkpoint and hfs.exists(output_path + '/_SUCCESS'):
        return None

    j = b.new_python_job(attributes={'name': f'vcf-to-mt/sample-group-{sample_group.sample_group_index}/{contig}',
                                     'task': 'vcf-to-mt'})
    j.cpu(cpu)
    j.image(docker)
    j.storage(storage)
    j.memory(memory)

    j.call(_vcf_to_mt, input_vcf, output_path, cpu)

    return j


def _union(mt_paths: hb.ResourceFile,
           output_path: str,
           billing_project: str,
           remote_tmpdir: str,
           regions: str,
           contig: str,
           n_partitions: int):
    import subprocess as sp
    import hail as hl
    from hail.backend.service_backend import ServiceBackend

    setup = f'''
hailctl config set batch/billing_project "{billing_project}"
hailctl config set batch/remote_tmpdir "{remote_tmpdir}"
hailctl config set batch/regions "{','.join(regions)}"
'''

    sp.run(setup, capture_output=True, shell=True, check=True)

    hl.init(backend="batch", app_name=f"union-{contig}")

    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)

    assert backend.regions == regions
    assert backend.remote_tmpdir == remote_tmpdir
    assert backend.billing_project == billing_project

    paths = []
    with open(mt_paths, 'r') as f:
        for line in f:
            path = line.rstrip("\n")
            paths.append(path)

    mt_init = hl.read_matrix_table(paths[0])
    intervals = mt_init._calculate_new_partitions(n_partitions)

    mt_left = hl.read_matrix_table(paths[0], _intervals=intervals)

    for idx, path in enumerate(paths[1:]):
        mt_right = hl.read_matrix_table(path, _intervals=intervals)
        mt_left = mt_left.union_cols(mt_right,
                                     drop_right_row_fields=True,
                                     row_join_type='outer')

    mt = mt_left

    if output_path.endswith('.mt'):
        mt.write(output_path)
        mt_count = hl.read_matrix_table(output_path)
        print(mt_count.count())
        print(mt.describe())
    else:
        assert output_path.endswith('.vcf.bgz')
        hl.export_vcf(mt, output_path, tabix=True)


def union_sample_groups_from_vcfs(b: hb.Batch,
                                  vcf_paths: hb.ResourceFile,
                                  output_path: str,
                                  docker: str,
                                  cpu: int,
                                  memory: str,
                                  storage: str,
                                  billing_project: str,
                                  remote_tmpdir: str,
                                  regions: List[str],
                                  use_checkpoints: bool,
                                  contig: str,
                                  n_partitions: int) -> Optional[Job]:
    if use_checkpoints:
        if output_path.endswith('.vcf.bgz') and hfs.exists(output_path):
            return None
        if output_path.endswith('.mt') and hfs.exists(output_path + '/_SUCCESS'):
            return None

    j = b.new_python_job(attributes={'name': f'union/{contig}'})
    j.cpu(cpu)
    j.image(docker)
    j.storage(storage)
    j.memory(memory)
    j.spot(False)
    j.call(_union, vcf_paths, output_path, billing_project, remote_tmpdir, regions, contig, n_partitions)

    return j
