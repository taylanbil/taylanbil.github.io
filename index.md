---
layout: taylandefault
---

<ul>
  {% for post in site.posts %}
    {% if post.categories %}
     <li>
       <a href="{{ post.url }}">{{ post.title }}</a> || {{ post.categories }} || {{ post.date | date: "%Y-%m-%d" }}
     </li>
    {% endif %}
  {% endfor %}
</ul>
