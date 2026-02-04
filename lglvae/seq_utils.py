import torch
from Bio import SeqIO

seq_code = {
    "G": 0,
    "A": 1,
    "V": 2,
    "L": 3,
    "I": 4,
    "F": 5,
    "P": 6,
    "S": 7,
    "T": 8,
    "Y": 9,
    "C": 10,
    "M": 11,
    "K": 12,
    "R": 13,
    "H": 14,
    "W": 15,
    "D": 16,
    "E": 17,
    "N": 18,
    "Q": 19,
    "-": 20,
    "X": 20,
    "Z": 20,
    "B": 20,
}


def one_hot_encode_fasta(
    fasta_file: str, alphabet: dict = seq_code, device: str = "cpu"
) -> torch.Tensor:
    """Encode FASTA sequences as one-hot tensors.

    Args:
        fasta_file: Path to FASTA file
        alphabet: Amino acid to index mapping
        device: Device for tensor

    Returns:
        One-hot tensor of shape (num_sequences, num_aa, seq_length)
        This matches TensorFlow convention with softmax over amino acids (dim=1)
    """

    loaded_seqs = list(SeqIO.parse(fasta_file, "fasta"))
    num_aa = len(set(item[1] for item in alphabet.items()))
    try:
        numeric_sequences = torch.tensor(
            [[seq_code[aa] for aa in seq.seq] for seq in loaded_seqs], device=device
        ).long()
    except ValueError:
        raise ValueError("Not all sequences are the same length (probably).")
    
    # one_hot produces (batch, seq_len, num_aa)
    one_hot_sequences = torch.nn.functional.one_hot(numeric_sequences, num_aa).float()

    # Transpose to (batch, num_aa, seq_len) to match TensorFlow convention
    # TensorFlow: Reshape((num_aa_type, dim_msa_vars)) with Softmax(axis=1)
    one_hot_sequences = one_hot_sequences.permute(0, 2, 1)

    return one_hot_sequences
