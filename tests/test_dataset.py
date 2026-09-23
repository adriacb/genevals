from pathlib import Path

from genevals.core.dataset import Dataset
from genevals.core.types import Sample


def test_load_save_jsonl(tmp_path: Path):
    ds = Dataset(
        [
            Sample(id="1", input="hi", reference="hello", tags={"lang": "en"}),
            Sample(id="2", input="hola", reference="hello", tags={"lang": "es"}),
        ],
        name="greetings",
    )
    path = tmp_path / "ds.jsonl"
    ds.save(path)
    loaded = Dataset.load(path)
    assert len(loaded) == 2
    assert loaded.samples[0].input == "hi"


def test_filter_by_tag():
    ds = Dataset(
        [
            Sample(id="1", input="a", tags={"lang": "en"}),
            Sample(id="2", input="b", tags={"lang": "es"}),
        ]
    )
    en_only = ds.with_tag("lang", "en")
    assert len(en_only) == 1
    assert en_only.samples[0].id == "1"


def test_load_missing_ids_from_list(tmp_path: Path):
    path = tmp_path / "ds.json"
    path.write_text('[{"input": "a"}, {"input": "b"}]', encoding="utf-8")
    ds = Dataset.load(path)
    assert [s.id for s in ds] == ["0", "1"]


def test_yaml_roundtrip(tmp_path: Path):
    ds = Dataset([Sample(id="1", input="a", reference="b")])
    path = tmp_path / "ds.yaml"
    ds.save(path)
    loaded = Dataset.load(path)
    assert loaded.samples[0].reference == "b"
