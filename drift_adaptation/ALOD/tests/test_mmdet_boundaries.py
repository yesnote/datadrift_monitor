import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(root: Path):
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob('*.py') if '__pycache__' not in path.parts)


def _imports_matching(path: Path, module_names):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_any(alias.name, module_names):
                    violations.append((node.lineno, 'import %s' % alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if _matches_any(module, module_names):
                violations.append((node.lineno, 'from %s import ...' % module))
    return violations


def _matches_any(module, module_names):
    return any(module == name or module.startswith(name + '.') for name in module_names)


class MMDetBoundaryTest(unittest.TestCase):
    def test_method_and_runner_support_code_do_not_import_mmdet(self):
        checked_roots = [
            ROOT / 'methods',
            ROOT / 'tools' / 'common',
            ROOT / 'tools' / 'run_active_learning.py',
            ROOT / 'configs' / 'catalog',
        ]
        offenders = []
        for root in checked_roots:
            for path in _python_files(root):
                for lineno, statement in _imports_matching(path, ['mmdet']):
                    rel_path = path.relative_to(ROOT)
                    offenders.append('%s:%d %s' % (rel_path, lineno, statement))

        self.assertEqual(
            offenders,
            [],
            msg='method and runner support code must not import mmdet:\n%s'
            % '\n'.join(offenders),
        )

    def test_mmdet_alod_extension_does_not_import_alod_layers(self):
        offenders = []
        for path in _python_files(ROOT / 'mmdet' / 'alod'):
            for lineno, statement in _imports_matching(path, ['methods', 'tools', 'configs']):
                rel_path = path.relative_to(ROOT)
                offenders.append('%s:%d %s' % (rel_path, lineno, statement))

        self.assertEqual(
            offenders,
            [],
            msg='mmdet.alod extension code must not import ALOD method/tool/config layers:\n%s'
            % '\n'.join(offenders),
        )


if __name__ == '__main__':
    unittest.main()
