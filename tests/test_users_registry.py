# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from smritilila import users_registry as ur


@pytest.mark.parametrize("content", ["", "not json", "[]", '"a string"', "5", "null", '{"users": 5}', '{"users": {"a": 1}}', '{"users": [1, "x", null]}'])
def test_a_file_that_is_not_a_list_of_users_is_no_users_and_never_an_error(tmp_path, content) -> None:
    """load_users promises "never an error that takes a request down": the builder's gate reads it on a guest's request, so a hand-edited file
    of the wrong shape (valid JSON that is not an object, a users value that is not a list) answers no users, as an unreadable one does."""
    f = tmp_path / "users.json"
    f.write_text(content, encoding="utf-8")
    assert ur.load_users(f) == []
    assert ur.may_build("anyone@example.com", f) is False


def test_one_bad_entry_does_not_lose_the_good_ones(tmp_path) -> None:
    f = tmp_path / "users.json"
    f.write_text(json.dumps({"users": [{"email": "A@Example.com", "builder": True, "added": "2026-10-06"}, {"email": 5}, {"email": None},
                                       "stray", {"builder": True}, {"email": "b@example.com"}]}), encoding="utf-8")
    assert [(u.email, u.builder) for u in ur.load_users(f)] == [("a@example.com", True), ("b@example.com", False)]
    assert ur.may_build("a@example.com", f) and not ur.may_build("b@example.com", f)


def test_a_missing_file_is_no_users(tmp_path) -> None:
    assert ur.load_users(tmp_path / "nothing.json") == []
