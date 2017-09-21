---
layout: taylandefault
---

<ul>
  {% for post in site.posts %}
    {% if post.categories %}
     <li>
       <a href="{{ post.url }}">{{ post.title }}</a> || {{ post.categories }}
     </li>
    {% endif %}
  {% endfor %}
</ul>
