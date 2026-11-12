"""The citation reference must expose the actual paper record without JavaScript."""
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


PUBLIC = Path(__file__).resolve().parents[1] / 'public'
BASE = 'https://embodied-agent-arena.github.io/embodied-agent-arena/'
VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
             'link', 'meta', 'param', 'source', 'track', 'wbr'}


class CitationPage(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.stack, self.ids, self.links, self.meta = [], [], [], {}
        self.canonical, self.titles, self.scripts, self.schemas = [], [], [], []
        self.citation_blocks, self.text, self.copy_buttons = [], [], []
        self.current_title = self.current_script = self.current_citation = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if 'href' in attrs:
            self.links.append((tag, attrs))
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical.append(attrs['href'])
        if tag == 'meta':
            key = attrs.get('name', attrs.get('property'))
            self.meta.setdefault(key, []).append(attrs.get('content', ''))
        if tag == 'title':
            self.current_title = []
        if tag == 'script':
            self.current_script = {'attrs': attrs, 'text': []}
        if tag == 'button' and 'data-copy-citation' in attrs:
            self.copy_buttons.append(attrs)
        if tag == 'pre' and attrs.get('id') == 'research-bibtex':
            self.current_citation = {'attrs': attrs, 'ancestors': list(self.stack), 'text': []}
        if tag not in VOID_TAGS:
            self.stack.append((tag, attrs))

    def handle_data(self, data):
        if self.current_title is not None:
            self.current_title.append(data)
        if self.current_script is not None:
            self.current_script['text'].append(data)
        else:
            self.text.append(data)
        if self.current_citation is not None:
            self.current_citation['text'].append(data)

    def handle_endtag(self, tag):
        if tag == 'title' and self.current_title is not None:
            self.titles.append(''.join(self.current_title))
            self.current_title = None
        if tag == 'script' and self.current_script is not None:
            script = self.current_script
            script['text'] = ''.join(script['text'])
            self.scripts.append(script)
            if script['attrs'].get('type') == 'application/ld+json':
                self.schemas.append(json.loads(script['text']))
            self.current_script = None
        if tag == 'pre' and self.current_citation is not None:
            block = self.current_citation
            block['text'] = ''.join(block['text'])
            self.citation_blocks.append(block)
            self.current_citation = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


class CitationAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (PUBLIC / 'citation.html').read_text()
        cls.page = CitationPage(cls.source)
        cls.csl = json.loads((PUBLIC / 'citation.csl.json').read_text())
        cls.bib = (PUBLIC / 'citation.bib').read_bytes()

    def test_visible_bibtex_and_plain_text_preserve_canonical_record(self):
        self.assertEqual((PUBLIC / 'citation.txt').read_bytes(), self.bib)
        self.assertEqual(len(self.page.citation_blocks), 1)
        block = self.page.citation_blocks[0]
        self.assertEqual(block['text'].encode('utf-8'), self.bib)
        self.assertNotIn('hidden', block['attrs'])
        self.assertEqual(block['attrs'].get('tabindex'), '0')
        self.assertTrue(all(tag != 'details' and 'hidden' not in attrs
                            for tag, attrs in block['ancestors']))
        self.assertIn('<!-- BEGIN PAPER BIBTEX -->', self.source)
        self.assertIn('<!-- END PAPER BIBTEX -->', self.source)

    def test_citation_actions_work_with_static_fallbacks(self):
        links = [attrs for tag, attrs in self.page.links if tag == 'a']
        self.assertTrue(any(a['href'] == 'citation.bib' and 'download' in a for a in links))
        self.assertTrue(any(a['href'] == 'citation.txt' and 'download' not in a for a in links))
        self.assertTrue(any(a['href'] == 'citation.csl.json' for a in links))
        self.assertEqual(len(self.page.copy_buttons), 1)
        self.assertEqual(self.page.copy_buttons[0]['data-copy-citation'], 'research-bibtex')
        self.assertIn('hidden', self.page.copy_buttons[0])
        self.assertIn('role="status" aria-live="polite"', self.source)
        self.assertIn(r'\cite{huang2026frontier}', ''.join(self.page.text))
        self.assertEqual([s['attrs'].get('src') for s in self.page.scripts if 'src' in s['attrs']],
                         ['research-pages.js?v=20261108'])

    def test_visible_identity_has_all_authors_and_version(self):
        text = ''.join(self.page.text)
        self.assertIn(self.csl['title'], text)
        self.assertEqual(len(self.csl['author']), 10)
        for author in self.csl['author']:
            self.assertIn(f'{author["given"]} {author["family"]}', text)
        self.assertIn(self.csl['DOI'], text)
        self.assertIn('1 October 2026', text)
        self.assertIn('23 October 2026', text)
        self.assertIn('arXiv v3', text)
        self.assertIn('the same Embodied Agent Arena paper', text)
        self.assertIn('they are not separate papers', text)

    def test_structured_data_identifies_paper_not_citation_as_fulltext(self):
        self.assertEqual(len(self.page.schemas), 1)
        page = self.page.schemas[0]
        self.assertEqual(page['@type'], 'WebPage')
        self.assertEqual(page['url'], BASE + 'citation.html')
        article = page['mainEntity']
        self.assertEqual(article['@type'], 'ScholarlyArticle')
        self.assertEqual(article['@id'], BASE + '#paper')
        self.assertEqual(article['name'], self.csl['title'])
        self.assertEqual(article['url'], self.csl['URL'])
        self.assertEqual(article['datePublished'], '2026-10-01')
        self.assertEqual(article['dateModified'], '2026-10-23')
        self.assertEqual(article['version'], 'arXiv v3 (2026-10-23)')
        self.assertEqual([a['name'] for a in article['author']],
                         [f'{a["given"]} {a["family"]}' for a in self.csl['author']])
        self.assertIn('https://doi.org/' + self.csl['DOI'], article['sameAs'])
        self.assertIn({'@type': 'PropertyValue', 'propertyID': 'DOI', 'value': self.csl['DOI']},
                      article['identifier'])
        self.assertNotIn('articleBody', article)
        self.assertNotIn('encoding', article)

    def test_canonical_head_and_static_links_resolve(self):
        self.assertEqual(self.page.canonical, [BASE + 'citation.html'])
        self.assertEqual(len(self.page.titles), 1)
        self.assertLessEqual(len(self.page.titles[0]), 60)
        self.assertEqual(len(self.page.meta['description']), 1)
        self.assertLessEqual(len(self.page.meta['description'][0]), 160)
        self.assertNotIn('noindex', ' '.join(self.page.meta.get('robots', [])))
        self.assertTrue({'paper', 'bibtex', 'evidence', 'research-bibtex'} <= set(self.page.ids))
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)))
        for _, attrs in self.page.links:
            url = urlsplit(attrs['href'])
            if url.scheme or url.netloc:
                continue
            target = PUBLIC / unquote(url.path or 'citation.html')
            if target.is_dir():
                target = target / 'index.html'
            with self.subTest(link=attrs['href']):
                self.assertTrue(target.is_file(), str(target))
                if url.fragment and target.suffix == '.html':
                    self.assertIn(unquote(url.fragment), CitationPage(target.read_text()).ids)

    def test_generator_propagates_changed_primary_bibtex_idempotently(self):
        root = PUBLIC.parent
        research_pages = ['public/' + name for name in
                          ('geoprobe.html', 'findings.html', 'research-topics.html', 'citation.html')]
        inputs = ['scripts/build_public_results.py', 'README.md', 'public/data/study.json',
                  'public/data/citation.bib', 'public/index.html', *research_pages]
        outputs = ['README.md', 'CITATION.cff', 'public/index.html', 'public/citation.bib',
                   'public/citation.txt', 'public/citation.csl.json',
                   'public/data/model-comparison.csv', *research_pages]
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            for relative in inputs:
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, destination)

            primary = temporary_root / 'public/data/citation.bib'
            original = primary.read_text()
            sentinel_note = r'note={Version 3, sync sentinel <record> \& source}'
            changed, count = re.subn(r'note=\{[^\n]*\}', lambda _: sentinel_note, original)
            self.assertEqual(count, 1, 'The fixture must change the primary BibTeX note.')
            self.assertNotEqual(changed, original)
            primary.write_text(changed)
            expected = changed.strip() + '\n'

            def run_generator():
                result = subprocess.run(
                    [sys.executable, str(temporary_root / 'scripts/build_public_results.py')],
                    cwd=temporary_root, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run_generator()
            for relative in ('public/citation.bib', 'public/citation.txt'):
                with self.subTest(export=relative):
                    self.assertEqual((temporary_root / relative).read_bytes(), expected.encode('utf-8'))
            for relative in research_pages:
                with self.subTest(inline=relative):
                    rendered = CitationPage((temporary_root / relative).read_text())
                    self.assertEqual(len(rendered.citation_blocks), 1)
                    self.assertEqual(rendered.citation_blocks[0]['text'], expected)

            homepage = (temporary_root / 'public/index.html').read_text()
            home_citation = re.search(r'<pre id="bibtex">(.*?)</pre>', homepage, re.S)
            self.assertIsNotNone(home_citation)
            self.assertEqual(html.unescape(home_citation.group(1)), expected.rstrip('\n'))
            readme = (temporary_root / 'README.md').read_text()
            readme_citation = re.search(
                r'<!-- BEGIN CITATION -->\n```bibtex\n(.*?)```\n<!-- END CITATION -->', readme, re.S)
            self.assertIsNotNone(readme_citation)
            self.assertEqual(readme_citation.group(1), expected)

            first_run = {relative: (temporary_root / relative).read_bytes() for relative in outputs}
            run_generator()
            self.assertEqual(primary.read_text(), changed)
            self.assertEqual(first_run,
                             {relative: (temporary_root / relative).read_bytes() for relative in outputs})


if __name__ == '__main__':
    unittest.main()
