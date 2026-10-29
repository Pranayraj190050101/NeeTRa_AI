"""Chapter catalog tests - the hierarchy must come from the index, never hardcode."""

from __future__ import annotations

from datetime import date

from learning.catalog import Chapter, ChapterCatalog, humanise
from learning.generation import GenieGenerator
from learning.motivation import daily_motivation


def test_humanise_reads_file_stems_like_a_student_does():
    assert humanise("04_moving_charges_and_magnetism") == "Moving Charges And Magnetism"
    assert humanise("plant_physiology.pdf") == "Plant Physiology"
    assert humanise("Chapter-5 Thermodynamics NCERT") == "Thermodynamics"
    assert humanise("dna_and_rna") == "DNA And RNA"


def test_catalog_groups_sources_by_subject_in_neet_order():
    catalog = ChapterCatalog.from_sources(
        [
            "biology/plant_physiology.pdf",
            "physics/mechanics.pdf",
            "chemistry/mole_concept.pdf",
            "general/revision.pdf",
        ]
    )
    assert catalog.subjects == ("physics", "chemistry", "biology", "general")
    assert [chapter.name for chapter in catalog.list("physics")] == ["Mechanics"]
    assert not catalog.is_empty


def test_catalog_subject_inference_falls_back_to_keywords():
    catalog = ChapterCatalog.from_sources(["misc/the_cell_cycle.pdf", "misc/unknown_topic.pdf"])
    by_name = {chapter.name: chapter for chapter in catalog.chapters}
    assert by_name["The Cell Cycle"].subject == "biology"  # keyword inference
    assert by_name["Unknown Topic"].subject == "general"  # honest bucket, no guess


def test_catalog_find_matches_label_file_and_source():
    catalog = ChapterCatalog.from_sources(["physics/mechanics.pdf"])
    for probe in ("Mechanics", "mechanics.pdf", "physics/mechanics.pdf", "  mechanics "):
        found = catalog.find(probe)
        assert isinstance(found, Chapter) and found.subject == "physics"
    assert catalog.find("photosynthesis") is None
    assert catalog.find("") is None


def test_catalog_counts_and_summaries():
    catalog = ChapterCatalog.from_sources(
        ["physics/mechanics.pdf", "physics/optics.pdf"],
        subject_chunks={"physics": 42, "chemistry": 0},
    )
    assert catalog.count_for("physics") == 2
    assert catalog.size_of("physics") == 42
    assert catalog.names("physics") == ["Mechanics", "Optics"]
    assert catalog.summary()["chapters"] == 2
    assert catalog.of("chemistry") == []


def test_catalog_from_a_real_index(vector_store):
    """The live FAISS index is the only source of the chapter list."""
    catalog = ChapterCatalog.from_manager(vector_store)
    assert sorted(chapter.name for chapter in catalog.chapters) == [
        "Mechanics",
        "Mole Concept",
        "Plant Physiology",
    ]
    assert catalog.size_of("physics") == 1
    assert ChapterCatalog.from_manager(None).is_empty  # type: ignore[arg-type]


def test_empty_catalog_is_reported_not_crashed():
    catalog = ChapterCatalog.empty()
    assert catalog.is_empty
    assert catalog.subjects == ()
    assert catalog.find("anything") is None
    assert bool(catalog) is False


def test_daily_motivation_is_stable_within_a_day_and_refreshes():
    day = date(2026, 10, 23)
    first = daily_motivation(on_date=day)
    assert first == daily_motivation(on_date=day)
    assert first.line
    assert first.day_of_year == day.timetuple().tm_yday
    assert daily_motivation(on_date=date(2026, 10, 24)).day_of_year == first.day_of_year + 1


def test_streak_unlocks_a_streak_line():
    # Day 296 of 2026 is even, which is the branch a live streak unlocks.
    assert daily_motivation(on_date=date(2026, 10, 23), streak_days=5).kind == "streak"
    assert daily_motivation(on_date=date(2026, 10, 23)).kind == "daily"


def test_generator_grounds_requests_in_the_index(settings, vector_store):
    generator = GenieGenerator(settings, vector_store=vector_store)
    grounding = generator.ground("mole concept avogadro", subject="chemistry", passages=2)
    assert grounding.grounded
    assert grounding.citations
    assert "chemistry" in grounding.context
    assert generator.index_ready()


def test_generator_prefers_the_requested_chapter_file(settings, vector_store):
    generator = GenieGenerator(settings, vector_store=vector_store)
    grounding = generator.ground(
        "photosynthesis chloroplast light reactions",
        subject="biology",
        passages=2,
        prefer_file="plant_physiology.pdf",
    )
    assert grounding.citations[0]["file_name"] == "plant_physiology.pdf"


def test_generator_degrades_gracefully_without_an_index(settings):
    generator = GenieGenerator(settings)  # no index built in tmp_path
    grounding = generator.ground("anything", subject="physics")
    assert not grounding.grounded
    assert grounding.notices  # the student is told why
    assert generator.ground("").notices
