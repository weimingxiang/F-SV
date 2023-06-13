import pysam
import torch

path = "/home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/sorted_final_merged.bam"

sam_file = pysam.AlignmentFile(path, "rb")

chr_list = sam_file.references
chr_length = sam_file.lengths
for chromosome, chr_len in zip(chr_list, chr_length):
    count = torch.zeros(9, dtype=torch.long)
    print(chromosome)
    print(count)
    for read in sam_file.fetch(chromosome):
        if read.is_unmapped or read.is_secondary:
            continue
        for operation, length in read.cigar:  # (operation, length)
            count[operation] += length
    print(count)
    torch.save(
        count, "/home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/count_op/" + chromosome + '.pt')

sam_file.close()
