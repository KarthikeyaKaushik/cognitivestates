"""Draw a saved behavioral diagram with Graphviz's force-directed layout.

Usage:
  python scripts/plot_behavior_diagram.py path/to/diagram.json
  python scripts/plot_behavior_diagram.py path/to/diagram.json --states 0 2 19 7 12

Requires Graphviz's `neato` executable. No third-party Python packages needed.
"""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

INPUT_COLORS = {
    'pop()': '#0072B2',
    'push(x)': '#D55E00',
    'push(y)': '#009E73',
}


def diagram_dot(machine, *, selected_states=None):
    """Keep arrows separate and label nodes with their inferred state IDs."""
    states = set(machine['states'])
    if machine['start_state'] not in states:
        raise ValueError('The start state is missing from the diagram states')
    visible = states if selected_states is None else states.intersection(selected_states)
    if selected_states is not None and (not visible or visible != set(selected_states)):
        raise ValueError('Select at least one state and use only IDs in the saved diagram')
    unknown_inputs = set(machine['input_tokens']) - INPUT_COLORS.keys()
    if unknown_inputs:
        raise ValueError(f'No input colors defined for {sorted(unknown_inputs)}')
    legend = ''.join(
        f'<TD><FONT COLOR="{INPUT_COLORS[token]}">{escape(token)}</FONT></TD>'
        for token in machine['input_tokens'])
    start_legend = ('           <TR><TD COLSPAN="3">Start state: diamond</TD></TR>'
                    if machine['start_state'] in visible else '')
    lines = [
        'digraph behavior {',
        '  graph [layout=neato, start=7, overlap=false, splines=true,',
        '         outputorder=edgesfirst, sep="+18", pad=0.3, dpi=220,',
        '         bgcolor=white, fontname="Helvetica", fontsize=12, labelloc=b,',
        '         label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="12">',
        f'           <TR>{legend}</TR>',
        '           <TR><TD COLSPAN="3">Input: edge color; edge label: output</TD></TR>',
        start_legend,
        '         </TABLE>>];',
        '  node [shape=circle, style=filled, fillcolor="#F1F1F1",',
        '        color="#333333", penwidth=1.3, width=0.38, height=0.38,',
        '        fontname="Helvetica", fontsize=12,',
        '        fixedsize=true];',
        '  edge [fontname="Helvetica", fontsize=11, penwidth=1.5,',
        '        arrowsize=0.65, len=1.35];',
    ]
    for state in sorted(visible):
        attributes = [f'label={json.dumps(str(state))}']
        if state == machine['start_state']:
            attributes.append('shape=diamond, width=0.48, height=0.48, '
                              'fillcolor="#CCCCCC"')
        lines.append(f'  "{state}" [{", ".join(attributes)}];')
    for transition in machine['transitions']:
        source, target = transition['state'], transition['next_state']
        if source not in states or target not in states:
            raise ValueError(f'Transition references an unknown state: {transition}')
        if source not in visible or target not in visible:
            continue
        token = transition['input']
        if token not in machine['input_tokens']:
            raise ValueError(f'Transition input is absent from the vocabulary: {token}')
        color = INPUT_COLORS[token]
        label = transition['output']
        lines.append(
            f'  "{source}" -> "{target}" '
            f'[color="{color}", fontcolor="{color}", label={json.dumps(label)}];')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def render_diagram(machine, directory, *, selected_states=None):
    """Save an editable DOT source plus PNG and vector SVG/PDF renderings."""
    neato = shutil.which('neato')
    if neato is None:
        raise RuntimeError(
            'Graphviz neato is required to draw the diagram. '
            'On macOS, install Graphviz with: brew install graphviz')
    selected_states = None if selected_states is None else set(selected_states)
    source = diagram_dot(machine, selected_states=selected_states)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = 'diagram' if selected_states is None else 'diagram_subset_' + '_'.join(
        str(state) for state in sorted(selected_states))
    dot_path = directory / f'{name}.dot'
    dot_path.write_text(source)
    command = [neato, str(dot_path)]
    paths = []
    for extension in ('svg', 'png', 'pdf'):
        path = directory / f'{name}.{extension}'
        command.extend([f'-T{extension}', '-o', str(path)])
        paths.append(path)
    subprocess.run(command, check=True)
    if selected_states is None:
        for extension in ('dot', 'svg', 'png', 'pdf'):
            (directory / f'diagram_traversal_counts.{extension}').unlink(missing_ok=True)
    for path in paths:
        print(f'Saved force-directed diagram to {path}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('diagram', type=Path, help='Saved diagram.json to draw.')
    parser.add_argument('--states', type=int, nargs='+',
                        help='Render only these states and transitions between them.')
    args = parser.parse_args()
    with args.diagram.open() as f:
        machine = json.load(f)
    if args.states is not None:
        unknown = set(args.states) - set(machine['states'])
        if unknown:
            parser.error(f'State IDs not in the saved diagram: {sorted(unknown)}')
    render_diagram(machine, args.diagram.parent, selected_states=args.states)


if __name__ == '__main__':
    main()
