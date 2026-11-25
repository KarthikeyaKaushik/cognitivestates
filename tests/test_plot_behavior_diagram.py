import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from scripts.plot_behavior_diagram import diagram_dot, render_diagram


class DiagramPlotTests(unittest.TestCase):
    def setUp(self):
        self.machine = dict(
            start_state=0, states=[0, 1, 20],
            input_tokens=['pop()', 'push(x)', 'push(y)'],
            transitions=[
                dict(state=0, input='pop()', output='n', next_state=1),
                dict(state=0, input='push(x)', output='x', next_state=1),
                dict(state=1, input='push(y)', output='e', next_state=0),
                dict(state=1, input='pop()', output='n', next_state=1)])

    def test_dot_preserves_each_transition_and_marks_start_only(self):
        dot = diagram_dot(self.machine)
        self.assertIn('layout=neato', dot)
        self.assertIn('start=7', dot)
        self.assertIn('"0" [label="0", shape=diamond', dot)
        self.assertIn('"1" [label="1"];', dot)
        self.assertIn('"20" [label="20"];', dot)
        self.assertEqual(dot.count(' -> '), 4)
        self.assertEqual(dot.count('"0" -> "1"'), 2)
        self.assertIn('"1" -> "1"', dot)
        for color in ('#0072B2', '#D55E00', '#009E73'):
            self.assertIn(f'color="{color}"', dot)
        self.assertIn('label="x"', dot)
        self.assertIn('label="e"', dot)

    def test_missing_graphviz_has_actionable_error(self):
        with patch('scripts.plot_behavior_diagram.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'brew install graphviz'):
                render_diagram(self.machine, Path('unused'))

    def test_subset_keeps_only_internal_edges_and_retains_isolated_nodes(self):
        dot = diagram_dot(self.machine, selected_states={1, 20})
        self.assertIn('"1" [label="1"];', dot)
        self.assertIn('"20" [label="20"];', dot)
        self.assertNotIn('"0" [', dot)
        self.assertNotIn('"0" ->', dot)
        self.assertNotIn('-> "0"', dot)
        self.assertNotIn('Start state: diamond', dot)
        self.assertEqual(dot.count(' -> '), 1)
        self.assertIn('"1" -> "1"', dot)
        with self.assertRaisesRegex(ValueError, 'saved diagram'):
            diagram_dot(self.machine, selected_states={1, 99})

    @unittest.skipUnless(shutil.which('neato'), 'Graphviz neato is not installed')
    def test_subset_render_uses_separate_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            full = directory / 'diagram.png'
            full.write_bytes(b'original')
            paths = render_diagram(self.machine, directory, selected_states={0, 1})
            self.assertCountEqual([path.name for path in paths],
                                  [f'diagram_subset_0_1.{ext}' for ext in ('svg', 'png', 'pdf')])
            self.assertEqual(full.read_bytes(), b'original')
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            groups = ET.parse(paths[0]).findall('.//svg:g', ns)
            self.assertCountEqual([g.find('svg:title', ns).text for g in groups
                                   if g.get('class') == 'node'], ['0', '1'])
            self.assertEqual(sum(g.get('class') == 'edge' for g in groups), 4)

    def test_invalid_start_or_transition_state_is_rejected(self):
        self.machine['start_state'] = 2
        with self.assertRaisesRegex(ValueError, 'start state'):
            diagram_dot(self.machine)
        self.machine['start_state'] = 0
        self.machine['transitions'][0]['next_state'] = 2
        with self.assertRaisesRegex(ValueError, 'unknown state'):
            diagram_dot(self.machine)

    @unittest.skipUnless(shutil.which('neato'), 'Graphviz neato is not installed')
    def test_rendered_diagram_has_state_ids_and_output_only_arrows(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            legacy = [directory / f'diagram_traversal_counts.{ext}'
                      for ext in ('dot', 'svg', 'png', 'pdf')]
            counts = directory / 'state_visit_counts.csv'
            for path in [*legacy, counts]:
                path.touch()
            paths = render_diagram(self.machine, directory)
            self.assertTrue(all(not path.exists() for path in legacy))
            self.assertTrue(counts.exists())
            self.assertCountEqual([path.name for path in paths],
                                  ['diagram.svg', 'diagram.png', 'diagram.pdf'])
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            root = ET.parse(Path(temp) / 'diagram.svg').getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            groups = root.findall('.//svg:g', ns)
            edges = [g for g in groups if g.get('class') == 'edge']
            nodes = [g for g in groups if g.get('class') == 'node']
            self.assertEqual(len(edges), 4)
            self.assertEqual(len(nodes), 3)
            arrow_paths = [edge.find('svg:path', ns).get('d') for edge in edges]
            self.assertEqual(len(set(arrow_paths)), 4)
            for node in nodes:
                self.assertEqual([text.text for text in node.findall('svg:text', ns)],
                                 [node.find('svg:title', ns).text])
            start = next(g for g in nodes if g.find('svg:title', ns).text == '0')
            self.assertIsNotNone(start.find('svg:polygon', ns))
            for edge, transition in zip(edges, self.machine['transitions']):
                self.assertEqual(edge.find('svg:text', ns).text, transition['output'])


if __name__ == '__main__':
    unittest.main()
