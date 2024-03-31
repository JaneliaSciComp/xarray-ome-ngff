import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("docs"), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample):
    if "test=skip" not in example.prefix_tags():
        eval_example.run_print_check(example)
