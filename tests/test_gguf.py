from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gguf_explorer.gguf import GgufFile, write_sample_gguf


class GgufParserTests(unittest.TestCase):
    def test_reads_sample_metadata_tree_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sample_gguf(Path(tmp) / "sample.gguf")
            reader = GgufFile(path)
            try:
                self.assertEqual(reader.version, 3)
                self.assertEqual(reader.tensor_count, 2)
                self.assertEqual(reader.alignment, 32)
                self.assertIn("blk.0.attn_q.weight", reader.tensors_by_name)
                self.assertIn("blk.0.ffn_down.weight", reader.tensors_by_name)

                tree = reader.tree()
                self.assertEqual(tree["children"][0]["name"], "blk")

                bf16 = reader.sample_tensor("blk.0.attn_q.weight", start=1, count=2)
                self.assertEqual(bf16["rows"][0]["decoded"], 1.0)
                self.assertEqual(bf16["rows"][1]["decoded"], -2.5)

                q8_static = reader.sample_tensor("blk.0.ffn_down.weight", start=0, count=3, mode="static")
                self.assertEqual([row["value"] for row in q8_static["rows"]], [-16, -15, -14])

                q8_final = reader.sample_tensor("blk.0.ffn_down.weight", start=0, count=3, mode="dequantized")
                self.assertEqual([row["decoded"] for row in q8_final["rows"]], [-4.0, -3.75, -3.5])

                q8_compare = reader.sample_tensor(
                    "blk.0.ffn_down.weight",
                    start=0,
                    count=3,
                    mode="dequantized",
                    reference=reader,
                )
                self.assertTrue(q8_compare["reference"]["compatible"])
                self.assertEqual([row["reference_value"] for row in q8_compare["rows"]], [-4.0, -3.75, -3.5])
                self.assertEqual([row["diff"] for row in q8_compare["rows"]], [0.0, 0.0, 0.0])
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
