import copy
from typing import List, Optional

import hailtop.batch as hb
import hailtop.batch_client.aioclient as bc
from hailtop.batch.job import Job
import hailtop.fs as hfs

from ..globals import Chunk, SampleGroup, get_bucket


def _vcf_to_mt(input_vcf: str, output_path: str):
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
              jg: hb.JobGroup,
              sample_group: SampleGroup,
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

    j = jg.new_python_job(attributes={'name': f'vcf-to-mt/sample-group-{sample_group.sample_group_index}/{contig}',
                                      'task': 'vcf-to-mt'})
    j.cpu(cpu)
    j.image(docker)
    j.storage(storage)
    j.memory(memory)

    j.call(_vcf_to_mt, input_vcf, output_path)

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
    sample_sizes = []
    with open(mt_paths, 'r') as f:
        for line in f:
            path, sample_size = line.rstrip("\n").split('\t')
            paths.append(path)
            sample_sizes.append(int(sample_size))


    def add_info_if_needed(mt):
        return mt.annotate_rows(info=mt.info.annotate(INFO=mt.info.get("INFO", hl.null(hl.tarray(hl.tfloat64)))))


    mt_init = hl.read_matrix_table(paths[0])
    intervals = mt_init._calculate_new_partitions(n_partitions)

    mt_left = hl.read_matrix_table(paths[0], _intervals=intervals)
    mt_left = add_info_if_needed(mt_left)
    mt_left = mt_left.annotate_rows(
        info=mt_left.info.annotate(N=sample_sizes[0], AF=mt_left.info.AF[0], INFO=mt_left.info.INFO[0],
                                   RAF=mt_left.info.RAF[0]))
    mt_left = mt_left.annotate_rows(**{"info_0": mt_left.info})

    for idx, path in enumerate(paths[1:]):
        mt_right = hl.read_matrix_table(path, _intervals=intervals)
        mt_right = add_info_if_needed(mt_right)
        mt_right = mt_right.annotate_rows(
            info=mt_right.info.annotate(N=sample_sizes[idx], AF=mt_right.info.AF[0], INFO=mt_right.info.INFO[0],
                                        RAF=mt_right.info.RAF[0]))
        mt_left = mt_left.union_cols(mt_right,
                                     drop_right_row_fields=False,
                                     row_join_type='outer')

    mt = mt_left

    n_samples = mt.count_cols()
    n_batches = len(paths)

    mt = mt.annotate_rows(info=mt.info.annotate(AF=hl.array([mt[f"info_{i}"].AF for i in range(n_batches)])))
    mt = mt.annotate_rows(info=mt.info.annotate(INFO=hl.array([mt[f"info_{i}"].INFO for i in range(n_batches)])))
    mt = mt.annotate_rows(info=mt.info.annotate(N=hl.array([mt[f"info_{i}"].N for i in range(n_batches)])))


    def GLIMPSE_AF(mt):
        return hl.sum(hl.map(lambda af, n: af * n, mt.info.AF, mt.info.N)) / n_samples


    def GLIMPSE_INFO(mt):
        return hl.if_else((GLIMPSE_AF(mt) == 0) | (GLIMPSE_AF(mt) == 1),
                          1,
                          1 - hl.sum(
                              hl.map(lambda af, n, info: (1 - info) * 2 * n * af * (1 - af), mt.info.AF, mt.info.N,
                                     mt.info.INFO)) / (2 * n_samples * GLIMPSE_AF(mt) * (1 - GLIMPSE_AF(mt))))


    mt = mt.annotate_rows(info=mt.info.annotate(AF=GLIMPSE_AF(mt), INFO=GLIMPSE_INFO(mt)))

    mt = mt.annotate_rows(info=mt.info.drop('N'))
    mt = mt.drop(*[f'info_{i}' for i in range(n_batches)])
    mt = mt.drop(*[f'rsid_{i}' for i in range(1, n_batches)])
    mt = mt.drop(*[f'qual_{i}' for i in range(1, n_batches)])
    mt = mt.drop(*[f'filters_{i}' for i in range(1, n_batches)])

    if output_path.endswith('.mt'):
        mt.write(output_path)
        mt_count = hl.read_matrix_table(output_path)
        print(mt_count.count())
        print(mt.describe())
    else:
        assert output_path.endswith('.vcf.bgz')
        hl.export_vcf(mt, output_path, tabix=True)


def union_sample_groups_from_vcfs(b: hb.Batch,
                                  jg: hb.JobGroup,
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

    j = jg.new_python_job(attributes={'name': f'union/{contig}'})
    j.cpu(cpu)
    j.image(docker)
    j.storage(storage)
    j.memory(memory)
    j.spot(False)
    j.call(_union, vcf_paths, output_path, billing_project, remote_tmpdir, regions, contig, n_partitions)

    return j
