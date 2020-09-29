import pytest
from datetime import datetime
from quantify.data.types import TUID


def test_TUID():
    tuid = TUID('20200409-123015-123-abcdef')

    dt = tuid.datetime()
    assert isinstance(dt, datetime)
    assert isinstance(tuid, str)

    assert dt.year == 2020
    assert dt.month == 4
    assert dt.day == 9

    assert dt.hour == 12
    assert dt.minute == 30
    assert dt.second == 15

    assert tuid.uuid() == 'abcdef'

    with pytest.raises(ValueError):
        tuid = TUID('200409-123015-123-abcdef')

    with pytest.raises(ValueError):
        tuid = TUID('200409-123015-123-abcdefasf')


def test_TUID_validation():
    TUID.is_valid('20200409-123015-123-abcdef')

    problems = [
        '20200409-123015-123-a',  # too short uid
        '20200409-123015-123-a135bcdefasf',  # too long uid
        '20200409-123015-abcdef',  # missing milliseconds
        '200409-123015-123-abcdef',  # 2 digit year
        '20200409123015-123-abcdef',  # missing dash
        "20200924-152319a414-131ece",  # wrong separator character
        "20200924-959399-414-131ece",  # impossible time
        "20200961-152319-414-131ece",  # impossible date
    ]

    for case in problems:
        with pytest.raises(ValueError):
            TUID.is_valid(case)
