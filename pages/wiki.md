---
layout: page
title: Wiki
description: 生活不再别处，脚下即是前方
keywords: 维基, Wiki
comments: false
menu: 维基
permalink: /wiki/
---

> 因为年轻，所以饱含希望;
  因为年轻，所以热泪盈眶;
  因为年轻，所以拥抱未来。

<ul class="listing">
{% for wiki in site.wiki %}
{% if wiki.title != "Wiki Template" %}
<li class="listing-item"><a href="{{ site.url }}{{ wiki.url }}">{{ wiki.title }}</a></li>
{% endif %}
{% endfor %}
</ul>
