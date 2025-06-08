import numpy as np
import random
from typing import List, Tuple, Optional


class LDPCCode:
    """
    LDPC (Low-Density Parity-Check) Code implementation
    """

    def __init__(self, n: int = 15, k: int = 11):
        """
        Initialize LDPC code
        Args:
            n: codeword length (total bits)
            k: information bits length
        """
        self.n = n  # Total length
        self.k = k  # Information length
        self.m = n - k  # Parity length

        # Generate parity check matrix H
        self.H = self._generate_parity_check_matrix()

        # Generate generator matrix G
        self.G = self._generate_generator_matrix()

    def _generate_parity_check_matrix(self) -> np.ndarray:
        """
        Generate a simple parity check matrix for demonstration
        In practice, this would be more sophisticated
        """
        # Create a simple structured LDPC matrix
        H = np.zeros((self.m, self.n), dtype=int)

        # Simple pattern for demonstration
        for i in range(self.m):
            # Each row has approximately 3-4 ones
            positions = [(i + j) % self.n for j in range(0, self.n, 3)][:4]
            for pos in positions:
                H[i, pos] = 1

        return H

    def _generate_generator_matrix(self) -> np.ndarray:
        """
        Generate generator matrix from parity check matrix
        Using systematic form: G = [I_k | P]
        """
        # Create identity matrix for systematic part
        I_k = np.eye(self.k, dtype=int)

        # Create parity matrix P (simplified approach)
        # In practice, this requires more sophisticated matrix operations
        P = np.random.randint(0, 2, (self.k, self.m))

        # Combine to form generator matrix
        G = np.hstack([I_k, P])

        return G

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

    def decode(self, received_bits: np.ndarray, max_iterations: int = 50) -> Tuple[np.ndarray, bool]:
        """
        Decode received bits using belief propagation algorithm
        Args:
            received_bits: Received (possibly corrupted) bits
            max_iterations: Maximum number of iterations
        Returns:
            Tuple of (decoded_bits, converged)
        """
        if len(received_bits) != self.n:
            raise ValueError(f"Received bits must be {self.n} bits long")

        # Initialize
        decoded = received_bits.copy()

        for iteration in range(max_iterations):
            # Check syndrome
            syndrome = np.dot(self.H, decoded) % 2

            # If syndrome is zero, decoding is successful
            if np.sum(syndrome) == 0:
                return decoded, True

            # Simple bit-flipping algorithm
            # Count parity check violations for each bit
            violations = np.dot(self.H.T, syndrome) % 2

            # Find bit with most violations
            max_violations = np.max(np.dot(self.H.T, syndrome))
            if max_violations > 0:
                # Flip bits with maximum violations
                candidates = np.where(np.dot(self.H.T, syndrome) == max_violations)[0]
                if len(candidates) > 0:
                    bit_to_flip = candidates[0]
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
        bits.extend([0] * (8 - len(bits) % 8))

    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte_value = int(''.join(map(str, byte_bits)), 2)
        chars.append(chr(byte_value))

    return ''.join(chars)


def add_noise(bits: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
    """
    Add random bit errors to simulate channel noise
    Args:
        bits: Original bits
        error_rate: Probability of bit error
    Returns:
        Bits with added errors
    """
    noisy_bits = bits.copy()
    n_errors = int(len(bits) * error_rate)

    # Randomly select positions to flip
    error_positions = random.sample(range(len(bits)), min(n_errors, len(bits)))

    for pos in error_positions:
        noisy_bits[pos] = 1 - noisy_bits[pos]

    print(f"Added {len(error_positions)} bit errors at positions: {error_positions}")
    return noisy_bits


def process_string_with_ldpc(text: str, error_rate: float = 0.1) -> None:
    """
    Complete pipeline: string -> bits -> LDPC encode -> add noise -> decode -> compare
    Args:
        text: Input text to process
        error_rate: Bit error rate to simulate
    """
    print(f"Original text: '{text}'")
    print("=" * 50)

    # Convert string to bits
    original_bits = string_to_bits(text)
    print(f"Original bits ({len(original_bits)} bits): {original_bits}")

    # Initialize LDPC codec
    ldpc = LDPCCode(n=15, k=11)

    # Process in chunks of k bits
    all_encoded = []
    all_original_chunks = []

    # Pad bits if necessary
    padded_bits = original_bits.copy()
    while len(padded_bits) % ldpc.k != 0:
        padded_bits.append(0)

    print(f"Padded bits ({len(padded_bits)} bits): {padded_bits}")

    # Encode chunks
    for i in range(0, len(padded_bits), ldpc.k):
        chunk = np.array(padded_bits[i:i + ldpc.k])
        all_original_chunks.append(chunk)
        encoded_chunk = ldpc.encode(chunk)
        all_encoded.extend(encoded_chunk)

    all_encoded = np.array(all_encoded)
    print(f"LDPC encoded ({len(all_encoded)} bits): {all_encoded.tolist()}")

    # Add noise
    print("\n" + "=" * 50)
    print("ADDING CHANNEL NOISE")
    print("=" * 50)
    noisy_bits = add_noise(all_encoded, error_rate)
    print(f"Corrupted bits: {noisy_bits.tolist()}")

    # Decode chunks
    print("\n" + "=" * 50)
    print("LDPC DECODING")
    print("=" * 50)

    decoded_chunks = []
    decode_success = []

    for i in range(0, len(noisy_bits), ldpc.n):
        received_chunk = noisy_bits[i:i + ldpc.n]
        decoded_chunk, success = ldpc.decode(received_chunk)
        decoded_chunks.append(decoded_chunk[:ldpc.k])  # Extract info bits
        decode_success.append(success)
        print(f"Chunk {i // ldpc.n + 1}: {'SUCCESS' if success else 'FAILED'}")

    # Reconstruct decoded bits
    decoded_info_bits = []
    for chunk in decoded_chunks:
        decoded_info_bits.extend(chunk)

    # Trim to original length
    decoded_info_bits = decoded_info_bits[:len(original_bits)]

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)

    # Convert back to strings
    try:
        original_string = bits_to_string(original_bits)
        decoded_string = bits_to_string(decoded_info_bits)

        print(f"Original text:  '{original_string}'")
        print(f"Decoded text:   '{decoded_string}'")

        # Calculate bit error rates
        original_array = np.array(original_bits)
        decoded_array = np.array(decoded_info_bits)

        bit_errors = np.sum(original_array != decoded_array)
        ber_before = np.sum(np.array(original_bits) != np.array(padded_bits[:len(original_bits)]))

        print(f"\nBit Error Analysis:")
        print(f"Bits before LDPC: {len(original_bits)}")
        print(f"Bits after encoding: {len(all_encoded)}")
        print(f"Remaining bit errors: {bit_errors}/{len(original_bits)}")
        print(f"Bit Error Rate: {bit_errors / len(original_bits):.3f}")
        print(f"String match: {'YES' if original_string == decoded_string else 'NO'}")

    except Exception as e:
        print(f"Error converting bits to string: {e}")
        print(f"Original bits:  {original_bits}")
        print(f"Decoded bits:   {decoded_info_bits}")


def demo_ldpc():
    """
    Demonstrate LDPC encoding/decoding with different scenarios
    """
    print("LDPC ENCODER/DECODER DEMONSTRATION")
    print("=" * 70)

    # Test cases
    test_strings = [
        "Hello",
        "LDPC",
        "Test123"
    ]

    error_rates = [0.05, 0.15, 0.25]

    for test_string in test_strings:
        for error_rate in error_rates:
            print(f"\n{'=' * 70}")
            print(f"TEST: '{test_string}' with {error_rate * 100}% error rate")
            print(f"{'=' * 70}")

            try:
                process_string_with_ldpc(test_string, error_rate)
            except Exception as e:
                print(f"Error processing '{test_string}': {e}")

            print("\n" + "-" * 70)


if __name__ == "__main__":
    # Run demonstration
    demo_ldpc()

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter text to encode/decode (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break

            error_rate = float(input("Enter error rate (0.0-1.0, default 0.1): ") or "0.1")
            process_string_with_ldpc(user_input, error_rate)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")