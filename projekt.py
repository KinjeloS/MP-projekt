import numpy as np
import random
from typing import List, Tuple, Optional


class LDPCCode:
    """
    LDPC (Low-Density Parity-Check) Code implementation
    """

    def __init__(self, n: int = 16, k: int = 8):
        """
        Initialize LDPC code
        Args:
            n: codeword length (total bits)
            k: information bits length
        """
        self.n = n  # Total length
        self.k = k  # Information length
        self.m = n - k  # Parity length

        # Generate matrices
        self.H, self.G = self._generate_matrices()

    def _generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate compatible parity check matrix H and generator matrix G
        Using systematic form where G = [I_k | P] and H = [P^T | I_m]
        """
        # Create random parity matrix P (k x m)
        P = np.random.randint(0, 2, (self.k, self.m))

        # Generator matrix in systematic form: G = [I_k | P]
        I_k = np.eye(self.k, dtype=int)
        G = np.hstack([I_k, P])

        # Parity check matrix in systematic form: H = [P^T | I_m]
        I_m = np.eye(self.m, dtype=int)
        H = np.hstack([P.T, I_m])

        return H, G

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits using LDPC
        Args:
            info_bits: Information bits to encode
        Returns:
            Encoded codeword
        """
        if len(info_bits) != self.k:
            raise ValueError(f"Input must be {self.k} bits long")

        # Matrix multiplication in GF(2)
        codeword = np.dot(info_bits, self.G) % 2
        return codeword.astype(int)

    def check_codeword(self, codeword: np.ndarray) -> bool:
        """
        Check if codeword is valid (syndrome = 0)
        """
        syndrome = np.dot(self.H, codeword) % 2
        return np.sum(syndrome) == 0

    def decode(self, received_bits: np.ndarray, max_iterations: int = 50) -> Tuple[np.ndarray, bool]:
        """
        Decode received bits using iterative decoding
        Args:
            received_bits: Received (possibly corrupted) bits
            max_iterations: Maximum number of iterations
        Returns:
            Tuple of (decoded_bits, converged)
        """
        if len(received_bits) != self.n:
            raise ValueError(f"Received bits must be {self.n} bits long")

        decoded = received_bits.copy().astype(int)

        for iteration in range(max_iterations):
            # Check syndrome
            syndrome = np.dot(self.H, decoded) % 2

            # If syndrome is zero, decoding is successful
            if np.sum(syndrome) == 0:
                return decoded, True

            # Simple majority-logic decoding
            # For each bit position, count how many parity checks it participates in that fail
            error_counts = np.zeros(self.n)

            for i in range(self.n):
                # Find which parity checks involve bit i
                parity_checks_for_bit_i = np.where(self.H[:, i] == 1)[0]
                # Count how many of these parity checks fail
                error_counts[i] = np.sum(syndrome[parity_checks_for_bit_i])

            # Find the bit position with maximum error count
            max_errors = np.max(error_counts)
            if max_errors > 0:
                # Flip the bit with most parity check failures
                candidates = np.where(error_counts == max_errors)[0]
                bit_to_flip = candidates[0]  # Choose first if tie
                decoded[bit_to_flip] = 1 - decoded[bit_to_flip]

        return decoded, False


def string_to_bits(text: str) -> List[int]:
    """
    Convert string to binary representation
    Args:
        text: Input string
    Returns:
        List of bits
    """
    bits = []
    for char in text:
        # Convert each character to 8-bit binary
        char_bits = format(ord(char), '08b')
        bits.extend([int(b) for b in char_bits])
    return bits


def bits_to_string(bits: List[int]) -> str:
    """
    Convert binary representation back to string
    Args:
        bits: List of bits
    Returns:
        Reconstructed string
    """
    if len(bits) % 8 != 0:
        # Pad with zeros if necessary
        bits = bits + [0] * (8 - len(bits) % 8)

    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        if len(byte_bits) == 8:  # Only process complete bytes
            byte_value = int(''.join(map(str, byte_bits)), 2)
            if byte_value > 0:  # Skip null characters from padding
                chars.append(chr(byte_value))

    return ''.join(chars)


def add_noise(bits: np.ndarray, error_rate: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    """
    Add random bit errors to simulate channel noise
    Args:
        bits: Original bits
        error_rate: Probability of bit error
    Returns:
        Tuple of (bits_with_errors, error_positions)
    """
    noisy_bits = bits.copy()
    n_errors = max(1, int(len(bits) * error_rate)) if error_rate > 0 else 0

    error_positions = []
    if n_errors > 0:
        # Randomly select positions to flip
        error_positions = random.sample(range(len(bits)), min(n_errors, len(bits)))

        for pos in error_positions:
            noisy_bits[pos] = 1 - noisy_bits[pos]

    return noisy_bits, error_positions


def process_string_with_ldpc(text: str, error_rate: float = 0.1) -> None:
    """
    Complete pipeline: string -> bits -> LDPC encode -> add noise -> decode -> compare
    Args:
        text: Input text to process
        error_rate: Bit error rate to simulate
    """
    print(f"Original text: '{text}'")
    print("=" * 60)

    # Convert string to bits
    original_bits = string_to_bits(text)
    print(f"Original bits ({len(original_bits)} bits):")
    print(f"  {original_bits}")

    # Initialize LDPC codec
    ldpc = LDPCCode(n=16, k=8)
    print(f"\nLDPC Parameters: n={ldpc.n}, k={ldpc.k}, m={ldpc.m}")

    # Process in chunks of k bits
    all_encoded = []
    all_original_chunks = []

    # Pad bits if necessary
    padded_bits = original_bits.copy()
    while len(padded_bits) % ldpc.k != 0:
        padded_bits.append(0)

    print(f"Padded bits ({len(padded_bits)} bits): {padded_bits}")

    # Verify encoding works correctly first
    print(f"\nENCODING PROCESS:")
    print("-" * 40)

    # Encode chunks
    for i, chunk_start in enumerate(range(0, len(padded_bits), ldpc.k)):
        chunk = np.array(padded_bits[chunk_start:chunk_start + ldpc.k])
        all_original_chunks.append(chunk)
        encoded_chunk = ldpc.encode(chunk)

        # Verify encoding
        is_valid = ldpc.check_codeword(encoded_chunk)
        print(f"Chunk {i + 1}: {chunk.tolist()} -> {encoded_chunk.tolist()} (Valid: {is_valid})")

        all_encoded.extend(encoded_chunk)

    all_encoded = np.array(all_encoded)
    print(f"\nTotal encoded bits ({len(all_encoded)}): {all_encoded.tolist()}")

    # Add noise
    print(f"\nADDING CHANNEL NOISE (Error rate: {error_rate * 100:.1f}%)")
    print("=" * 60)

    noisy_bits, error_positions = add_noise(all_encoded, error_rate)

    if error_positions:
        print(f"Added {len(error_positions)} bit errors at positions: {error_positions}")
        print(f"Original:  {all_encoded.tolist()}")
        print(f"Corrupted: {noisy_bits.tolist()}")

        # Show differences
        diff_str = ""
        for i, (orig, corr) in enumerate(zip(all_encoded, noisy_bits)):
            if orig != corr:
                diff_str += f"Pos {i}: {orig}->{corr} "
        print(f"Changes: {diff_str}")
    else:
        print("No errors added (0% error rate)")
        print(f"Bits: {noisy_bits.tolist()}")

    # Decode chunks
    print(f"\nLDPC DECODING")
    print("=" * 60)

    decoded_chunks = []
    decode_results = []

    for i, chunk_start in enumerate(range(0, len(noisy_bits), ldpc.n)):
        received_chunk = noisy_bits[chunk_start:chunk_start + ldpc.n]
        print(f"\nDecoding chunk {i + 1}:")
        print(f"  Received: {received_chunk.tolist()}")

        decoded_chunk, success = ldpc.decode(received_chunk)
        info_bits = decoded_chunk[:ldpc.k]  # Extract information bits

        decoded_chunks.append(info_bits)
        decode_results.append(success)

        print(f"  Decoded:  {decoded_chunk.tolist()}")
        print(f"  Info bits: {info_bits.tolist()}")
        print(f"  Status: {'SUCCESS' if success else 'FAILED'}")

        # Verify decoded codeword
        is_valid_after = ldpc.check_codeword(decoded_chunk)
        print(f"  Valid codeword: {is_valid_after}")

    # Reconstruct decoded bits
    decoded_info_bits = []
    for chunk in decoded_chunks:
        decoded_info_bits.extend(chunk.tolist())

    # Trim to original length
    decoded_info_bits = decoded_info_bits[:len(original_bits)]

    # Extract corrupted information bits (what data would look like without correction)
    corrupted_info_bits = []
    for i, chunk_start in enumerate(range(0, len(noisy_bits), ldpc.n)):
        corrupted_chunk = noisy_bits[chunk_start:chunk_start + ldpc.n]
        # Extract just the information bits from corrupted codeword (systematic code)
        corrupted_info = corrupted_chunk[:ldpc.k].tolist()
        corrupted_info_bits.extend(corrupted_info)

    # Trim to original length
    corrupted_info_bits = corrupted_info_bits[:len(original_bits)]

    print(f"\nFINAL COMPARISON")
    print("=" * 60)

    print(f"Original bits:    {original_bits}")
    print(f"Corrupted bits:   {corrupted_info_bits}")
    print(f"LDPC decoded:     {decoded_info_bits}")

    # Convert back to strings
    try:
        original_string = bits_to_string(original_bits)
        corrupted_string = bits_to_string(corrupted_info_bits)
        decoded_string = bits_to_string(decoded_info_bits)

        print(f"\nSTRING COMPARISON:")
        print(f"Original text:     '{original_string}'")
        print(f"Without correction: '{corrupted_string}'")
        print(f"With LDPC correction: '{decoded_string}'")

        # Calculate statistics
        corrupted_bit_errors = sum(1 for a, b in zip(original_bits, corrupted_info_bits) if a != b)
        corrected_bit_errors = sum(1 for a, b in zip(original_bits, decoded_info_bits) if a != b)
        total_bits = len(original_bits)

        print(f"\nERROR ANALYSIS:")
        print(
            f"  Bit errors without correction: {corrupted_bit_errors}/{total_bits} ({corrupted_bit_errors / total_bits * 100:.1f}%)")
        print(
            f"  Bit errors after LDPC:        {corrected_bit_errors}/{total_bits} ({corrected_bit_errors / total_bits * 100:.1f}%)")
        print(f"  Bits corrected by LDPC:       {corrupted_bit_errors - corrected_bit_errors}")

        print(f"\nTEXT RECOVERY:")
        print(f"  Without correction: {'READABLE' if original_string == corrupted_string else 'CORRUPTED'}")
        print(
            f"  With LDPC:          {'PERFECT' if original_string == decoded_string else 'PARTIAL' if corrected_bit_errors < corrupted_bit_errors else 'FAILED'}")
        print(f"  LDPC decoding:      {'SUCCESS' if all(decode_results) else 'PARTIAL SUCCESS'}")

        # Show character-by-character comparison if strings are short
        if len(original_string) <= 10:
            print(f"\nCHARACTER-BY-CHARACTER:")
            max_len = max(len(original_string), len(corrupted_string), len(decoded_string))
            orig_padded = original_string.ljust(max_len)
            corr_padded = corrupted_string.ljust(max_len)
            ldpc_padded = decoded_string.ljust(max_len)

            print(f"  Pos: ", end="")
            for i in range(max_len):
                print(f"{i:>3}", end="")
            print()

            print(f"  Orig: ", end="")
            for char in orig_padded:
                print(f"'{char}'", end="")
            print()

            print(f"  Corr: ", end="")
            for char in corr_padded:
                print(f"'{char}'", end="")
            print()

            print(f"  LDPC: ", end="")
            for char in ldpc_padded:
                print(f"'{char}'", end="")
            print()

            print(f"  Match:", end="")
            for i in range(max_len):
                orig_char = orig_padded[i] if i < len(orig_padded) else ' '
                corr_char = corr_padded[i] if i < len(corr_padded) else ' '
                ldpc_char = ldpc_padded[i] if i < len(ldpc_padded) else ' '

                if orig_char == ldpc_char:
                    print(" ✓ ", end="")
                elif orig_char == corr_char:
                    print(" - ", end="")
                else:
                    print(" ✗ ", end="")
            print()

    except Exception as e:
        print(f"Error converting bits to string: {e}")
        print(f"Original bits:    {original_bits}")
        print(f"Corrupted bits:   {corrupted_info_bits}")
        print(f"LDPC decoded:     {decoded_info_bits}")


def test_basic_ldpc():
    """
    Test basic LDPC functionality
    """
    print("BASIC LDPC TEST")
    print("=" * 50)

    ldpc = LDPCCode(n=16, k=8)

    # Test with simple data
    test_data = np.array([1, 0, 1, 1, 0, 1, 0, 1])  # 8 bits
    print(f"Test data: {test_data}")

    # Encode
    encoded = ldpc.encode(test_data)
    print(f"Encoded: {encoded}")
    print(f"Valid codeword: {ldpc.check_codeword(encoded)}")

    # Test decoding without errors
    decoded, success = ldpc.decode(encoded)
    print(f"Decoded (no errors): {decoded}")
    print(f"Success: {success}")
    print(f"Info bits match: {np.array_equal(test_data, decoded[:8])}")

    # Test with one error
    corrupted = encoded.copy()
    corrupted[2] = 1 - corrupted[2]  # Flip bit 2
    print(f"\nCorrupted (bit 2 flipped): {corrupted}")
    print(f"Valid codeword: {ldpc.check_codeword(corrupted)}")

    decoded_corr, success_corr = ldpc.decode(corrupted)
    print(f"Decoded (with error): {decoded_corr}")
    print(f"Success: {success_corr}")
    print(f"Info bits match: {np.array_equal(test_data, decoded_corr[:8])}")


def demo_ldpc():
    """
    Demonstrate LDPC encoding/decoding with different scenarios
    """
    print("LDPC ENCODER/DECODER DEMONSTRATION")
    print("=" * 70)

    # First, basic test
    test_basic_ldpc()

    print("\n" + "=" * 70)
    print("STRING PROCESSING TESTS")
    print("=" * 70)

    # Test cases
    test_cases = [
        ("Hi", 0.0),  # No errors
        ("Hi", 0.01),  # 10% errors
        ("Test", 0.02),  # 15% errors
        ("LDPC", 0.05),  # 20% errors
    ]

    for text, error_rate in test_cases:
        print(f"\n{'=' * 70}")
        print(f"TEST: '{text}' with {error_rate * 100:.0f}% error rate")
        print(f"{'=' * 70}")

        try:
            process_string_with_ldpc(text, error_rate)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run demonstration
    demo_ldpc()

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter text to test LDPC encoding/decoding")
    print("(Press Ctrl+C to exit)")

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if not user_input:
                continue

            error_rate_input = input("Enter error rate (0.0-0.5, default 0.1): ").strip()
            try:
                error_rate = float(error_rate_input) if error_rate_input else 0.1
                error_rate = max(0.0, min(0.5, error_rate))  # Clamp to reasonable range
            except ValueError:
                error_rate = 0.1

            process_string_with_ldpc(user_input, error_rate)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()