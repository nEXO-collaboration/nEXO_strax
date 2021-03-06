"""Create datastructure documentation page

This will add a page with various svg graphs and html tables
describing the datastructure: dependencies, columns provided,
and configuration options that apply to each plugins.

For extra credit, the SVGs are clickable.
"""
from collections import defaultdict
import os
import shutil

import pandas as pd
import graphviz
import strax
import nEXO_strax

this_dir = os.path.dirname(os.path.realpath(__file__))


page_header = """
Straxen datastructure
========================

This page is an autogenerated reference for all the plugins in nEXO_strax's
`xenon1t_analysis` context. 

Colors indicate data kinds. To load tables with different data kinds,
you currently need more than one `get_df` (or `get_array`) commands.

"""


template = """
{data_type}
--------------------------------------------------------

Description
~~~~~~~~~~~~~~~~~~~~~~

Provided by plugin: {p.__class__.__name__}

Data kind: {kind}

{docstring}


Columns provided
~~~~~~~~~~~~~~~~~~~~~~
.. raw:: html

{columns}


Dependencies
~~~~~~~~~~~~~~~~~~~~~~
.. raw:: html

{svg}


Configuration options
~~~~~~~~~~~~~~~~~~~~~~~

These are all options that affect this data type. 
This also includes options taken by dependencies of this datatype,
because changing any of those options affect this data indirectly.

.. raw:: html

{config_options}


"""


kind_colors = dict(
    events='#ffffff',
    peaks='#98fb98',
    records='#ffa500',
    raw_records='#ff4500')


def add_spaces(x):
    """Add four spaces to every line in x

    This is needed to make html raw blocks in rst format correctly
    """
    y = ''
    if isinstance(x, str):
        x = x.split('\n')
    for q in x:
        y += '    ' + q
    return y


def build_datastructure_doc():
    out = page_header

    pd.set_option('display.max_colwidth', -1)

    st = strax.Context(
        register_all=[x
                      for x in nEXO_strax.contexts.common_opts['register_all']
                      if x != nEXO_strax.cuts])

    # Too lazy to write proper graph sorter
    # Make dictionary {total number of dependencies below -> list of plugins}
    plugins_by_deps = defaultdict(list)
    for pn, p in st._plugin_class_registry.items():
        plugins = st._get_plugins((pn,), run_id='0')
        plugins_by_deps[len(plugins)].append(pn)

    os.makedirs(this_dir + '/graphs', exist_ok=True)

    for n_deps in list(reversed(sorted(list(plugins_by_deps.keys())))):
        for data_type in plugins_by_deps[n_deps]:
            plugins = st._get_plugins((data_type,), run_id='0')

            # Create dependency graph
            g = graphviz.Digraph(format='svg')
            # g.attr('graph', autosize='false', size="25.7,8.3!")
            for d, p in plugins.items():
                g.node(d,
                       style='filled',
                       href='#' + d.replace('_', '-'),
                       fillcolor=kind_colors.get(p.data_kind_for(d), 'grey'))
                for dep in p.depends_on:
                    g.edge(d, dep)

            fn = this_dir + '/graphs/' + data_type
            g.render(fn)
            with open(fn + '.svg', mode='r') as f:
                svg = add_spaces(f.readlines()[5:])

            config_df = st.show_config(d).sort_values(by='option')

            # Shorten long default values
            config_df['default'] = [
                x[:10] + '...' + x[-10:]
                if isinstance(x, str) and len(x) > 30 else x
                for x in config_df['default'].values]

            p = plugins[data_type]

            out += template.format(
                p=p,
                svg=svg,
                data_type=data_type,
                columns=add_spaces(
                    st.data_info(data_type).to_html(index=False)
                ),
                kind=p.data_kind_for(data_type),
                docstring=p.__doc__ if p.__doc__ else '(no plugin description)',
                config_options=add_spaces(
                    config_df.to_html(index=False))
            )

    with open(this_dir + '/reference/datastructure.rst', mode='w') as f:
        f.write(out)

    shutil.rmtree(this_dir + '/graphs')


try:
    if __name__ == '__main__':
        build_datastructure_doc()
except KeyError:
    # Whatever
    pass
