# -*- coding: utf-8 -*-
# Copyright (C) 2026 iamsad3508
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of Baige (百格附着力分析). Commercial licence: see COMMERCIAL.md
"""服务端面向用户的文案（zh / en）。

请求级语言：HTTP 头 ``X-Baige-Lang``（前端 post()/uploadXHR 统一带，取值 zh|en）。
用 contextvars 而不是全局变量，因为分析都跑在 threadpool 里——anyio 会复制上下文，
同一个进程里两种语言的请求不会互相串。

用法::

    from srvi18n import L
    lines.append(L("合计 %5.2f%%", "Total %5.2f%%", pct))

``L(zh, en, *args)`` 的 en 可以省（省了就中英都用中文），args 走 %-格式。
"""

import contextvars

_LANG = contextvars.ContextVar("baige_lang", default="zh")

SUPPORTED = ("zh", "en")


def set_lang(value):
    """把请求头值规整成 zh / en（认 en-US、en_GB 这类写法；其它语言一律当英文）。"""
    v = (value or "").strip().lower().replace("_", "-")
    v = v.split(",")[0].split(";")[0].split("-")[0]
    _LANG.set("en" if v and v != "zh" else "zh")


def lang():
    return _LANG.get()


def is_en():
    return _LANG.get() == "en"


def L(zh, en=None, *args):
    """按当前语言取模板；带 args 时套 %-格式（格式失败原样返回模板，不让文案炸掉分析）。"""
    tpl = en if (en is not None and _LANG.get() == "en") else zh
    if not args:
        return tpl
    try:
        return tpl % args
    except Exception:
        return tpl
