from __future__ import annotations

import random
import string
import unittest
import unicodedata

from rogueprompt import transforms as T


class CompatibilityTests(unittest.TestCase):
    def test_legacy_word_split_still_works(self) -> None:
        self.assertEqual(T.split_even_odd_words("alpha  beta\ngamma"), ("alpha gamma", "beta"))


class SegmentationTests(unittest.TestCase):
    def test_toy_segmentation_and_partition(self) -> None:
        spans = T.segment_text("bring blue pens today")
        self.assertEqual(spans, ("bring ", "blue ", "pens ", "today"))
        self.assertEqual(
            T.partition_even_odd(spans),
            (("bring ", "pens "), ("blue ", "today")),
        )

    def test_leading_trailing_and_mixed_whitespace(self) -> None:
        text = "\t  alpha  beta\n\ngamma\t"
        normalized = unicodedata.normalize("NFC", text)
        spans = T.segment_text(text)
        self.assertEqual(spans[0], "\t  ")
        self.assertEqual("".join(spans), normalized)

    def test_nfc_normalization(self) -> None:
        text = "Cafe\u0301 noir"
        self.assertEqual("".join(T.segment_text(text)), "Café noir")

    def test_empty_and_whitespace_only_rejected(self) -> None:
        for text in ("", " ", "\t\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    T.segment_text(text)
                with self.assertRaises(ValueError):
                    T.generate_rogueprompt(text)


class SerializationTests(unittest.TestCase):
    def test_round_trip_with_delimiter_characters(self) -> None:
        spans = ("a|b:c;d ", "||::;;\n", "é🙂")
        serialized = T.serialize_spans(spans)
        self.assertEqual(T.deserialize_spans(serialized), spans)

    def test_empty_stream_round_trip(self) -> None:
        self.assertEqual(T.serialize_spans(()), "")
        self.assertEqual(T.deserialize_spans(""), ())

    def test_malformed_serializations_rejected(self) -> None:
        for serialized in ("x:a", "2:a", "1:a|", "1:a;1:b", "3:ab"):
            with self.subTest(serialized=serialized):
                with self.assertRaises(ValueError):
                    T.deserialize_spans(serialized)


class CipherAndPayloadTests(unittest.TestCase):
    def test_vigenere_case_non_ascii_and_key_advancement(self) -> None:
        plaintext = "Ab-é Z!"
        encrypted = T.vigenere_encrypt(plaintext, "BC")
        self.assertEqual(encrypted, "Bd-é A!")
        self.assertEqual(T.vigenere_decrypt(encrypted, "BC"), plaintext)

    def test_toy_vector(self) -> None:
        spans = T.segment_text("bring blue pens today")
        even, odd = T.partition_even_odd(spans)
        serialized_even = T.serialize_spans(even)
        serialized_odd = T.serialize_spans(odd)
        encrypted_odd = T.vigenere_encrypt(serialized_odd, "LIME")
        payload = T.assemble_payload(serialized_even, encrypted_odd, "LIME")
        encoded = T.apply_rot13(payload)

        self.assertEqual(serialized_even, "6:bring |5:pens ")
        self.assertEqual(serialized_odd, "5:blue |5:today")
        self.assertEqual(encrypted_odd, "5:mtgi |5:ewpej")
        self.assertEqual(
            payload,
            "EVEN=6:bring |5:pens ;ODD=5:mtgi |5:ewpej;KEY=LIME;ORDER=0-even",
        )
        self.assertEqual(
            encoded,
            "RIRA=6:oevat |5:craf ;BQQ=5:zgtv |5:rjcrw;XRL=YVZR;BEQRE=0-rira",
        )
        self.assertEqual(T.reconstruct_payload(payload), "bring blue pens today")

    def test_outer_parser_uses_lengths_not_delimiter_search(self) -> None:
        text = "alpha;ODD=inside | beta;KEY=inside gamma;ORDER=inside"
        prompt = T.generate_rogueprompt(text, key="LIME")
        self.assertEqual(T.reconstruct_rogueprompt(prompt), text)

    def test_payload_parser_rejects_wrong_order(self) -> None:
        with self.assertRaises(ValueError):
            T.parse_payload("EVEN=1:a;ODD=;KEY=LIME;ORDER=1-odd")


class FullRoundTripTests(unittest.TestCase):
    CASES = (
        "bring blue pens today",
        "one",
        "one two",
        "one two three",
        " repeated   spaces  stay ",
        "\tleading\tand\nnewlines\n",
        "punctuation: | ; : stays attached!",
        "Café naïve 東京 🙂 text",
        "MiXeD CaSe ASCII",
    )

    def test_complete_round_trips(self) -> None:
        for text in self.CASES:
            with self.subTest(text=repr(text)):
                expected = unicodedata.normalize("NFC", text)
                prompt = T.generate_rogueprompt(text)
                self.assertEqual(T.reconstruct_rogueprompt(prompt), expected)

    def test_random_round_trips(self) -> None:
        rng = random.Random(1337)
        alphabet = string.ascii_letters + string.digits + " |:;,.!?é🙂\t\n"
        checked = 0
        while checked < 500:
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 180)))
            if not text.strip():
                continue
            self.assertEqual(
                T.reconstruct_rogueprompt(T.generate_rogueprompt(text, key="LIME")),
                unicodedata.normalize("NFC", text),
            )
            checked += 1

    def test_split_ablation_payloads_preserve_normalized_text(self) -> None:
        text = "  alpha | beta\n gamma;delta  "
        expected = unicodedata.normalize("NFC", text)

        no_rot13 = T.generate_no_rot13_prompt(text, key="LIME")
        payload = no_rot13.rsplit(T.PLAIN_PAYLOAD_MARKER, 1)[1].strip()
        self.assertEqual(T.reconstruct_payload(payload), expected)

        no_vigenere = T.generate_no_vigenere_prompt(text)
        payload = T.apply_rot13(
            no_vigenere.rsplit(T.ENCODED_PAYLOAD_MARKER, 1)[1].strip()
        )
        self.assertEqual(T.reconstruct_payload(payload, odd_is_encrypted=False), expected)

        splitting_only = T.generate_splitting_only_prompt(text)
        payload = splitting_only.rsplit(T.PLAIN_PAYLOAD_MARKER, 1)[1].strip()
        self.assertEqual(T.reconstruct_payload(payload, odd_is_encrypted=False), expected)


if __name__ == "__main__":
    unittest.main()
