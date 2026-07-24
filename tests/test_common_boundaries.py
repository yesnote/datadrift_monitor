import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(root: Path):
    return sorted(path for path in root.rglob('*.py') if '__pycache__' not in path.parts)


def _forbidden_tools_common_imports(path: Path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'tools.common' or alias.name.startswith('tools.common.'):
                    violations.append((node.lineno, 'import %s' % alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module == 'tools.common' or module.startswith('tools.common.'):
                violations.append((node.lineno, 'from %s import ...' % module))
    return violations


class CommonBoundaryTest(unittest.TestCase):
    def test_methods_do_not_import_tools_common(self):
        offenders = []
        for path in _python_files(ROOT / 'methods'):
            for lineno, statement in _forbidden_tools_common_imports(path):
                rel_path = path.relative_to(ROOT)
                offenders.append('%s:%d %s' % (rel_path, lineno, statement))

        self.assertEqual(
            offenders,
            [],
            msg='methods code must not import tools.common:\n%s' % '\n'.join(offenders),
        )


if __name__ == '__main__':
    unittest.main()
