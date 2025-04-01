import argparse
import asyncio
import base64
import json
import os
from shlex import quote as shq

from hailtop.aiotools.copy import copy_from_dict
import hailtop.batch as hb
from hailtop.utils import secret_alnum_string


async def submit(args):
    tmpdir_path_prefix = secret_alnum_string()

    remote_tmpdir = args['batch_remote_tmpdir']
    billing_project = args['billing_project']
    regions = args['batch_regions'].split(',')

    def cloud_prefix(path):
        return f'{remote_tmpdir}/{tmpdir_path_prefix}/{os.path.basename(path)}'

    manifest_cloud_file = cloud_prefix(args['reference_manifest'])

    backend = hb.ServiceBackend(billing_project=billing_project, regions=regions, remote_tmpdir=remote_tmpdir)

    if args['batch_id'] is not None:
        b = hb.Batch.from_batch_id(args['batch_id'], backend=backend)
    else:
        b = hb.Batch(name=args['batch_name'], backend=backend)

    j = b.new_bash_job(name='submit-jobs')
    j.image(args['docker_hail'])

    await copy_from_dict(
        files=[
            {'from': args['reference_manifest'], 'to': manifest_cloud_file}
        ]
    )

    local_manifest = '/reference_manifest.tsv'
    args['reference_manifest'] = local_manifest

    arguments_str = base64.b64encode(json.dumps(args).encode('utf-8')).decode('utf-8')

    manifest_input = b.read_input(manifest_cloud_file)

    j.command(f'mv {manifest_input} {local_manifest}')
    j.command(f'python3 -m glimpse_hail_batch.local_ancestry.local_ancestry "{shq(arguments_str)}"')

    batch_handle = await b._async_run(wait=False, disable_progress_bar=True)
    assert batch_handle
    print(f'Submitted batch {batch_handle.id}')

    await backend.async_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--billing-project', type=str, required=False)
    parser.add_argument('--batch-remote-tmpdir', type=str, required=False)
    parser.add_argument('--batch-regions', type=str, default="us-central1")
    parser.add_argument('--batch-name', type=str, default='local-ancestry')
    parser.add_argument('--batch-id', type=int, required=False)

    parser.add_argument('--seed', type=int, required=False, default=2423413432)

    parser.add_argument('--docker-flare', type=str, required=True)
    parser.add_argument('--docker-hail', type=str, required=True)

    parser.add_argument('--reference-manifest', type=str, required=True)
    parser.add_argument('--contig-col', type=str, required=True)
    parser.add_argument('--reference-file-col', type=str, required=True)
    parser.add_argument('--map-file-col', type=str, required=True)

    parser.add_argument('--reference-panel', type=str, required=True)

    parser.add_argument('--n-samples', type=int, required=False)

    parser.add_argument('--sample-group-size', type=int, required=True, default=100)

    parser.add_argument('--flare-cpu', type=int, required=True)
    parser.add_argument('--flare-memory', type=str, required=False, default='standard')
    parser.add_argument('--flare-storage', type=str, required=False)

    parser.add_argument('--mt-to-vcf-cpu', type=int, required=True)
    parser.add_argument('--mt-to-vcf-memory', type=str, required=False, default='standard')
    parser.add_argument('--mt-to-vcf-storage', type=str, required=False)

    parser.add_argument('--vcf-to-mt-cpu', type=int, required=True)
    parser.add_argument('--vcf-to-mt-memory', type=str, required=False, default='standard')
    parser.add_argument('--vcf-to-mt-storage', type=str, required=False)

    parser.add_argument('--union-sample-groups-cpu', type=int, required=True)
    parser.add_argument('--union-sample-groups-memory', type=str, required=False, default='standard')
    parser.add_argument('--union-sample-groups-storage', type=str, required=False)

    parser.add_argument('--contig', type=str, required=False)
    parser.add_argument('--sample-group-index', type=int, required=False)

    parser.add_argument('--input-file-dir', type=str, required=True)
    parser.add_argument('--input-file-regex', type=str, required=True)

    parser.add_argument('--staging-remote-tmpdir', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)

    parser.add_argument('--use-checkpoints', action='store_true', required=False)
    parser.add_argument('--save-checkpoints', action='store_true', required=False)

    parser.add_argument('--gcs-requester-pays-configuration', type=str, required=False)

    args = vars(parser.parse_args())

    print('submitting jobs with the following parameters:')
    print(json.dumps(args, indent=4))

    asyncio.run(submit(args))
