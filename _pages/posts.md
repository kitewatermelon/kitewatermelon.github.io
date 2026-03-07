---
title: "Posts"
layout: default
permalink: /posts/
author_profile: true
sidebar:
  nav: "categories"
---

<style>
.tag-filter {
  margin-bottom: 20px;
}
.tag-btn {
  display: inline-block;
  padding: 5px 12px;
  margin: 3px;
  border: 1px solid #ddd;
  border-radius: 15px;
  font-size: 0.8em;
  cursor: pointer;
  background: #f5f5f5;
  color: #333;
  transition: all 0.2s;
}
.tag-btn:hover {
  background: #007acc;
  color: white;
  border-color: #007acc;
}
.tag-btn.active {
  background: #007acc;
  color: white;
  border-color: #007acc;
}
.tag-group {
  margin-bottom: 10px;
}
.tag-group-title {
  font-weight: bold;
  font-size: 0.85em;
  color: #666;
  margin-right: 10px;
}
.post-item {
  padding: 15px 0;
  border-bottom: 1px solid #eee;
}
.post-item.hidden {
  display: none;
}
.post-title {
  font-size: 1.1em;
  margin-bottom: 5px;
}
.post-title a {
  color: #333;
  text-decoration: none;
}
.post-title a:hover {
  color: #007acc;
}
.post-meta {
  font-size: 0.8em;
  color: #888;
  margin-bottom: 8px;
}
.post-tags .tag {
  display: inline-block;
  padding: 2px 8px;
  margin: 2px;
  background: #f0f0f0;
  border-radius: 10px;
  font-size: 0.75em;
  color: #555;
}
.post-excerpt {
  font-size: 0.9em;
  color: #666;
  margin-top: 8px;
}
.filter-info {
  padding: 10px;
  background: #f9f9f9;
  border-radius: 5px;
  margin-bottom: 15px;
  font-size: 0.9em;
}
.clear-filter {
  color: #007acc;
  cursor: pointer;
  margin-left: 10px;
}
.search-box {
  width: 100%;
  padding: 10px 15px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 1em;
}
.search-box:focus {
  outline: none;
  border-color: #007acc;
}
</style>

<div id="main" role="main">
{% include sidebar.html %}

<div class="archive">
<h1 class="page__title">Posts</h1>

<!-- 검색창 -->
<input type="text" class="search-box" id="searchInput" placeholder="검색어를 입력하세요..." onkeyup="filterPosts()">

<!-- 태그 필터 -->
<div class="tag-filter">
  <div class="tag-group">
    <span class="tag-group-title">Category:</span>
    <span class="tag-btn" data-tag="Paper-Review" onclick="toggleTag(this)">Paper-Review</span>
    <span class="tag-btn" data-tag="Code-Review" onclick="toggleTag(this)">Code-Review</span>
    <span class="tag-btn" data-tag="Study" onclick="toggleTag(this)">Study</span>
  </div>
  <div class="tag-group">
    <span class="tag-group-title">Domain:</span>
    <span class="tag-btn" data-tag="Medical-AI" onclick="toggleTag(this)">Medical-AI</span>
    <span class="tag-btn" data-tag="Computer-Vision" onclick="toggleTag(this)">Computer-Vision</span>
  </div>
  <div class="tag-group">
    <span class="tag-group-title">Method:</span>
    <span class="tag-btn" data-tag="Contrastive-Learning" onclick="toggleTag(this)">Contrastive-Learning</span>
    <span class="tag-btn" data-tag="Self-Supervised-Learning" onclick="toggleTag(this)">Self-Supervised-Learning</span>
    <span class="tag-btn" data-tag="Foundation-Model" onclick="toggleTag(this)">Foundation-Model</span>
    <span class="tag-btn" data-tag="Wavelet" onclick="toggleTag(this)">Wavelet</span>
    <span class="tag-btn" data-tag="Segmentation" onclick="toggleTag(this)">Segmentation</span>
  </div>
  <div class="tag-group">
    <span class="tag-group-title">Conference:</span>
    <span class="tag-btn" data-tag="MICCAI" onclick="toggleTag(this)">MICCAI</span>
    <span class="tag-btn" data-tag="ICLR" onclick="toggleTag(this)">ICLR</span>
    <span class="tag-btn" data-tag="ICML" onclick="toggleTag(this)">ICML</span>
    <span class="tag-btn" data-tag="NIPS" onclick="toggleTag(this)">NIPS</span>
    <span class="tag-btn" data-tag="ECCV" onclick="toggleTag(this)">ECCV</span>
  </div>
</div>

<!-- 필터 정보 -->
<div class="filter-info" id="filterInfo" style="display:none;">
  선택된 태그: <span id="selectedTags"></span>
  <span class="clear-filter" onclick="clearAllTags()">[모두 해제]</span>
</div>

<!-- 포스트 목록 -->
<div id="postList">
{% assign sorted_posts = site.posts | sort: 'date' | reverse %}
{% for post in sorted_posts %}
<div class="post-item" data-categories="{{ post.categories | join: ',' }}" data-tags="{{ post.tags | join: ',' }}" data-title="{{ post.title | downcase }}" data-content="{{ post.excerpt | strip_html | downcase }}">
  <div class="post-title">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </div>
  <div class="post-meta">
    {{ post.date | date: "%Y-%m-%d" }}
  </div>
  <div class="post-tags">
    {% for category in post.categories %}
      <span class="tag">{{ category }}</span>
    {% endfor %}
    {% for tag in post.tags %}
      <span class="tag">{{ tag }}</span>
    {% endfor %}
  </div>
  {% if post.excerpt %}
  <div class="post-excerpt">{{ post.excerpt | strip_html | truncate: 150 }}</div>
  {% endif %}
</div>
{% endfor %}
</div>

</div>
</div>

<script>
let activeTags = new Set();

function toggleTag(element) {
  const tag = element.dataset.tag;

  if (activeTags.has(tag)) {
    activeTags.delete(tag);
    element.classList.remove('active');
  } else {
    activeTags.add(tag);
    element.classList.add('active');
  }

  filterPosts();
  updateFilterInfo();
}

function clearAllTags() {
  activeTags.clear();
  document.querySelectorAll('.tag-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  filterPosts();
  updateFilterInfo();
}

function updateFilterInfo() {
  const filterInfo = document.getElementById('filterInfo');
  const selectedTags = document.getElementById('selectedTags');

  if (activeTags.size > 0) {
    filterInfo.style.display = 'block';
    selectedTags.textContent = Array.from(activeTags).join(', ');
  } else {
    filterInfo.style.display = 'none';
  }
}

function filterPosts() {
  const searchInput = document.getElementById('searchInput').value.toLowerCase();
  const posts = document.querySelectorAll('.post-item');

  posts.forEach(post => {
    const categories = post.dataset.categories || '';
    const tags = post.dataset.tags || '';
    const title = post.dataset.title || '';
    const content = post.dataset.content || '';
    const allTags = categories + ',' + tags;

    // 태그 필터 체크
    let tagMatch = true;
    if (activeTags.size > 0) {
      tagMatch = Array.from(activeTags).every(tag =>
        allTags.toLowerCase().includes(tag.toLowerCase())
      );
    }

    // 검색어 필터 체크
    let searchMatch = true;
    if (searchInput) {
      searchMatch = title.includes(searchInput) ||
                    content.includes(searchInput) ||
                    allTags.toLowerCase().includes(searchInput);
    }

    // 둘 다 만족해야 표시
    if (tagMatch && searchMatch) {
      post.classList.remove('hidden');
    } else {
      post.classList.add('hidden');
    }
  });
}

// URL 파라미터로 태그 필터 적용
document.addEventListener('DOMContentLoaded', function() {
  const urlParams = new URLSearchParams(window.location.search);
  const tagParam = urlParams.get('tag');

  if (tagParam) {
    const tagBtn = document.querySelector(`.tag-btn[data-tag="${tagParam}"]`);
    if (tagBtn) {
      toggleTag(tagBtn);
    }
  }
});
</script>
