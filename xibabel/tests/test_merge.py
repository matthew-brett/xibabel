""" Test dictionary merge code.
"""

from xibabel.xutils import merge


def test_merge():
    a = {"name": "john", "phone":"123123123",
     "owns": {"cars": "Car 1", "motorbikes": "Motorbike 1"}}
    b = {"name": "john", "phone":"123", "owns": {"cars": "Car 2"}}
    assert merge(a, b) == {
        "name": "john", "phone": "123",
        "owns": {"cars": "Car 2", "motorbikes": "Motorbike 1"}}
    a = {"name": "john", "phone":"123123123",
         "owns": {"cars": {"Car 1": 1000, "Car 2": 2000},
                  "motorbikes": {"Motorbike 1": 100}}}
    b = {"name": "john", "phone":"123", "other": 1,
         "owns": {"cars": {"Car 1": 10000, "Car 4": 3000},
                  "motorbikes": {"Motorbike 2": 200}}}
    assert merge(a, b) == {
        "name": "john", "phone": "123", "other": 1,
        "owns": {"cars": {"Car 1": 10000, "Car 2": 2000, "Car 4": 3000},
                 "motorbikes": {"Motorbike 1": 100, "Motorbike 2": 200}}}
    b = {"name": "james", "phone": {'mobile': "+1 510 277 8554"}, "other": 1,
         "owns": ['Car 1', 'Car 2']}
    assert merge(a, b) == {
        "name": "james", "phone": {'mobile': "+1 510 277 8554"}, "other": 1,
         "owns": ['Car 1', 'Car 2']}
