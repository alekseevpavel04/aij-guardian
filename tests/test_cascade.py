from guardian.cascade import closest_to_cut, combine, merge, top_share


def test_top_share_labels_the_highest_scores():
    assert top_share([0.1, 0.9, 0.5, 0.7], 0.5) == [0, 1, 0, 1]


def test_unscored_rows_are_never_labelled_and_do_not_count():
    assert top_share([None, 0.9, 0.2, None], 0.5) == [0, 1, 0, 0]
    assert top_share([None, None], 0.5) == [0, 0]


def test_ties_at_the_cut_go_in():
    assert top_share([0.8, 0.8, 0.8, 0.1], 0.25) == [1, 1, 1, 0]


def test_rules_are_added_to_the_judge():
    assert combine([1, 0, 0], [0, 0, 1]) == [1, 0, 1]


def test_rows_nearest_the_cut_come_first():
    # ranks by score: 3 (0.9), 1 (0.7), 0 (0.5), 2 (0.1); the cut at share 0.5 falls after rank 1
    assert closest_to_cut([0.5, 0.7, 0.1, 0.9], 0.5)[:2] == [0, 1]
    assert closest_to_cut([None, 0.2], 0.5) == [1]


def test_reasoned_scores_replace_quick_ones_where_present():
    assert merge([0.4, 0.6, None], [0.99, None, 0.01]) == [0.99, 0.6, 0.01]
