from typing import List

import hailtop.fs as hfs


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

    def flare_output_file_root(self, contig: str):
        return f'{self.temp_dir}/flare/{contig}'

    def model_file(self):
        return f'{self.temp_dir}/flare/model'

    def flare_mt(self, contig: str):
        return f'{self.temp_dir}/flare-mt/{contig}.mt'

    def write_samples_list(self) -> str:
        with hfs.open(self.samples_list_file, 'w') as f:
            for sample_id in self.samples:
                f.write(f'{sample_id}\n')
        return self.samples_list_file

    def get_flare_output_file_names(self, contigs: List[str]) -> List[str]:
        output_files = []
        for contig in contigs:
            output_files.append(self.flare_output_file_root(contig))
        return output_files

    def get_flare_mts(self, contigs: List[str]) -> List[str]:
        output_files = []
        for contig in contigs:
            output_files.append(self.flare_mt(contig))
        return output_files


def split_samples_into_groups(samples: List[str],
                              desired_group_size: int,
                              output_dir: str) -> List[FlareSampleGroup]:
    groups: List[FlareSampleGroup] = []
    working_group = []
    sample_group_idx = 0
    for sample in samples:
        if len(working_group) >= desired_group_size:
            groups.append(FlareSampleGroup(working_group, sample_group_idx, output_dir))
            working_group = []
            sample_group_idx += 1
        working_group.append(sample)

    groups.append(FlareSampleGroup(working_group, sample_group_idx, output_dir))

    return groups
