"""Research pages must be usable without JavaScript and cite one published version."""
import json
import re
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'public'
BASE = 'https://embodied-agent-arena.github.io/embodied-agent-arena/'


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.meta, self.schemas = [], [], {}, []
        self.canonical, self.script, self.text = [], None, []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'meta':
            key = attrs.get('name', attrs.get('property'))
            self.meta.setdefault(key, []).append(attrs.get('content', ''))
        if tag in ('a', 'link') and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical.append(attrs['href'])
        if tag == 'script':
            self.script = {'type': attrs.get('type'), 'text': ''}

    def handle_data(self, data):
        if self.script is not None:
            self.script['text'] += data
        else:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.script is not None:
            if self.script['type'] == 'application/ld+json':
                self.schemas.append(json.loads(self.script['text']))
            self.script = None


class DiscoveryTests(unittest.TestCase):
    def test_new_pages_are_crawlable_and_links_resolve(self):
        for name in ('geoprobe.html', 'findings.html'):
            page = Page(PUBLIC / name)
            self.assertEqual(page.canonical, [BASE + name])
            self.assertEqual(len(page.ids), len(set(page.ids)))
            self.assertTrue(page.meta['description'][0])
            self.assertNotIn('noindex', ' '.join(page.meta.get('robots', [])))
            self.assertTrue(page.schemas)
            self.assertIn('citation.bib', page.links)
            self.assertIn('citation.csl.json', page.links)
            for link in page.links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                target = PUBLIC / unquote(url.path or name)
                if target.is_dir():
                    target = target / 'index.html'
                with self.subTest(page=name, link=link):
                    self.assertTrue(target.is_file(), str(target))
                    if url.fragment and target.suffix == '.html':
                        self.assertIn(unquote(url.fragment), Page(target).ids)

    def test_citations_and_scholar_metadata_identify_same_paper(self):
        study = json.loads((PUBLIC / 'data/study.json').read_text())
        page = Page(PUBLIC / 'index.html')
        csl = json.loads((PUBLIC / 'citation.csl.json').read_text())
        bib = (PUBLIC / 'citation.bib').read_text()
        paper = f'https://arxiv.org/abs/{study["arxiv"]}v{study["arxiv_version"]}'
        self.assertEqual(bib, (PUBLIC / 'data/citation.bib').read_text())
        self.assertEqual(csl['title'], study['title'])
        self.assertEqual(csl['URL'], paper)
        self.assertEqual(page.meta['citation_title'], [study['title']])
        self.assertEqual(page.meta['citation_author'], [a['name'] for a in study['authors']])
        self.assertEqual(page.meta['citation_doi'], [csl['DOI']])
        self.assertIn(csl['DOI'], bib)
        self.assertIn(paper, bib)
        self.assertIn(paper, (ROOT / 'CITATION.cff').read_text())
        article = next(s for s in page.schemas if s['@type'] == 'ScholarlyArticle')
        self.assertEqual(article['sameAs'], paper)
        self.assertEqual(article['dateModified'], study['manuscript_date'])
        self.assertEqual(len(article['author']), len(csl['author']))
        for path in ('README.md', 'public/data/study.json', 'public/index.html'):
            content = (ROOT / path).read_text().lower()
            self.assertNotIn('awaiting announcement', content)
            self.assertNotIn('announcement is pending', content)

    def test_sitemap_uses_canonical_pages_and_valid_dates(self):
        ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap = ET.parse(PUBLIC / 'sitemap.xml')
        urls = sitemap.findall('s:url', ns)
        locations = [u.findtext('s:loc', namespaces=ns) for u in urls]
        self.assertEqual(len(locations), len(set(locations)))
        for url in urls:
            location = url.findtext('s:loc', namespaces=ns)
            self.assertTrue(location.startswith(BASE))
            path = PUBLIC / (location[len(BASE):] or 'index.html')
            self.assertEqual(Page(path).canonical, [location])
            self.assertRegex(url.findtext('s:lastmod', namespaces=ns), r'^2026-\d{2}-\d{2}$')
        self.assertIn(BASE + 'geoprobe.html', locations)
        self.assertIn(BASE + 'findings.html', locations)

    def test_root_robots_allows_search_and_advertises_sitemap(self):
        project_robots = (PUBLIC / 'robots.txt').read_text()
        root_robots = (ROOT / 'deployment/github-root/robots.txt').read_text()
        self.assertEqual(root_robots, project_robots)
        self.assertIn('User-agent: *\nAllow: /', root_robots)
        self.assertIn('Sitemap: ' + BASE + 'sitemap.xml', root_robots)
        self.assertNotRegex(root_robots, r'Disallow:\s*/')

    def test_findings_keep_population_counts_in_static_text(self):
        page = Page(PUBLIC / 'findings.html')
        text = re.sub(r'\s+', ' ', ''.join(page.text))
        planning = json.loads((PUBLIC / 'data/report/planning-outcomes.json').read_text())
        for field in ('atLeastOneSucceeds', 'allFail', 'allSucceed'):
            self.assertIn(f'{planning[field]} / {planning["nTasks"]}', text)
        self.assertTrue({'planning-complementarity', 'execution-recovery', 'failure-token-cost'} <= set(page.ids))
        self.assertIn('RGB-expanded', text)
        self.assertIn('100', text)
        self.assertIn('65', text)
        self.assertIn('62', text)
        self.assertIn('55.5%', text)

    def test_homepage_exposes_research_before_case_browser(self):
        source = (PUBLIC / 'index.html').read_text()
        self.assertLess(source.index('id="research-summary"'), source.index('id="cases"'))
        for link in ('geoprobe.html', 'findings.html', 'citation.bib', 'citation.csl.json'):
            self.assertIn(link, Page(PUBLIC / 'index.html').links)


if __name__ == '__main__':
    unittest.main()
