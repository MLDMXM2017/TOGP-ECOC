#!/usr/bin/python2
# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 10:09
# @Author: Hanrui Wang
# @Target: To save some global variables, like training set and so on.


def _init():
    global _global_dict
    _global_dict = {}


def set_val(key, value):
    _global_dict[key] = value


def get_val(key, defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
