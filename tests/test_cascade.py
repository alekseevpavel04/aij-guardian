from guardian.cascade import combine, top_share


def test_top_share_labels_the_highest_scores():
    assert top_share([0.1, 0.9, 0.5, 0.7], 0.5) == [0, 1, 0, 1]


def test_unscored_rows_are_never_labelled_and_do_not_count():
    assert top_share([None, 0.9, 0.2, None], 0.5) == [0, 1, 0, 0]
    assert top_share([None, None], 0.5) == [0, 0]


def test_ties_at_the_cut_go_in():
    assert top_share([0.8, 0.8, 0.8, 0.1], 0.25) == [1, 1, 1, 0]


def test_rules_are_added_to_the_judge():
    assert combine([1, 0, 0], [0, 0, 1]) == [1, 0, 1]
