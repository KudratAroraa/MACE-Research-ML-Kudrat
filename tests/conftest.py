# provides one session-level reproduction result for slow regression tests
import pytest

from mace.reproductionPipeline import runFullPaperReproduction


# run the complete faithful reproduction once for all slow tests
@pytest.fixture(scope="session")
def paperResults():
    return runFullPaperReproduction(includeBruteForce=True)
