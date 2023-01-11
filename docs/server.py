#!/usr/bin/env python
from livereload import Server, shell

server = Server()
server.watch("docs/conf.py", shell("make html", cwd="docs"))
server.watch("docs/**/*.rst", shell("make html", cwd="docs"))
server.watch("docs/**/*.md", shell("make html", cwd="docs"))
server.watch("ogstools/**/*.py", shell("make html", cwd="docs"))
server.serve(root="docs/_build/html")
