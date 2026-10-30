
from beartype import beartype
from copy import deepcopy
from pathlib import Path
import numpy as np
import tqdm
import re

@beartype
def read_statements(vnnlib_filename: Path):
    with open(vnnlib_filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    assert len(lines) > 0

    open_parentheses = 0
    statements = []
    current_statement = ''

    for line in lines:
        comment_index = line.find(';')

        if comment_index != -1:
            line = line[:comment_index].rstrip()

        if not line:
            continue

        new_open = line.count('(')
        new_close = line.count(')')

        open_parentheses += new_open - new_close

        assert open_parentheses >= 0, "mismatched parenthesis in vnnlib file"

        current_statement += ' ' if current_statement else ''
        current_statement += line

        if open_parentheses == 0:
            statements.append(current_statement)
            current_statement = ''

    if current_statement:
        statements.append(current_statement)

    statements = [" ".join(s.split()) for s in statements]

    statements = [s.replace('( ', '(') for s in statements]

    statements = [s.replace(') ', ')') for s in statements]

    return statements


def update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs):

    if first.startswith("X_"):
        index = int(first[2:])

        assert not second.startswith("X") and not second.startswith("Y"), \
            f"input constraints must be box ({op} {first} {second})"
        assert 0 <= index < num_inputs, print(index, num_inputs)

        limits = rv_tuple[0][index]

        if op == "<=":
            limits[1] = min(float(second), limits[1])
        else:
            limits[0] = max(float(second), limits[0])

        assert limits[0] <= limits[1], f"{first} range is empty: {limits}"

    else:
        if op == ">=":
            first, second = second, first

        row = [0.0] * num_outputs
        rhs = 0.0

        if first.startswith("Y_") and second.startswith("Y_"):
            index1 = int(first[2:])
            index2 = int(second[2:])

            row[index1] = 1
            row[index2] = -1
        elif first.startswith("Y_"):
            index1 = int(first[2:])
            row[index1] = 1
            rhs = float(second)
        else:
            assert second.startswith("Y_")
            index2 = int(second[2:])
            row[index2] = -1
            rhs = -1 * float(first)

        mat, rhs_list = rv_tuple[1], rv_tuple[2]
        mat.append(row)
        rhs_list.append(rhs)


def make_input_box_dict(num_inputs):

    rv = {i: [-np.inf, np.inf] for i in range(num_inputs)}

    return rv


@beartype
def read_vnnlib(vnnlib_filename: str, regression: bool = False) -> list:
    return _read_vnnlib(vnnlib_filename=vnnlib_filename, regression=regression, mismatch_input_output=True)

@beartype
def _read_vnnlib(vnnlib_filename: str, regression: bool = False, mismatch_input_output: bool = False) -> list:
    vnnlib_filename = Path(vnnlib_filename)
    assert vnnlib_filename.is_file() and vnnlib_filename.suffix == ".vnnlib", vnnlib_filename

    regex_declare = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

    comparison_str = r"\((<=|>=) (\S+) (\S+)\)"

    dnf_clause_str = r"\(and\s*(" + comparison_str + r")+\)"

    regex_simple_assert = re.compile(r"^\(assert " + comparison_str + r"\)$")

    regex_dnf = re.compile(r"^\(assert \(or (" + dnf_clause_str + r")+\)\)$")

    lines = read_statements(vnnlib_filename)

    num_inputs = num_outputs = 0
    if mismatch_input_output:
        for line in lines:
            num = re.findall(r'X_(\d+)', line)
            if len(num):
                num_inputs = max(num_inputs, int(num[0]) + 1)
            num = re.findall(r'Y_(\d+)', line)
            if len(num):
                num_outputs = max(num_outputs, int(num[0]) + 1)
    else:
        for line in lines:
            declare = regex_declare.findall(line)
            if len(declare) == 0:
                continue
            elif len(declare) > 1:
                raise ValueError(f'There cannot be more than one declaration in one line: {line}')
            else:
                declare = declare[0]
                if declare[0] == 'X':
                    num_inputs = max(num_inputs, int(declare[1]) + 1)
                elif declare[0] == 'Y':
                    num_outputs = max(num_outputs, int(declare[1]) + 1)
                else:
                    raise ValueError(f'Unknown declaration: {line}')


    rv = []
    rv.append((make_input_box_dict(num_inputs), [], []))

    if regression:
        assert len(lines) == 3

    for line in lines:
        if len(regex_declare.findall(line)) > 0:
            continue

        groups = regex_simple_assert.findall(line)

        if groups:
            assert len(groups[0]) == 3, f"groups was {groups}: {line}"
            op, first, second = groups[0]

            for rv_tuple in rv:
                update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs)

            continue

        groups = regex_dnf.findall(line)
        if not groups:
            print(f"[VNNLIB] Skipped parsing line: {line}.")
            continue

        tokens = line.replace("(", " ").replace(")", " ").split()
        tokens = tokens[2:]

        conjuncts = " ".join(tokens).split("and")[1:]

        if regression:
            cases = []
            for c  in conjuncts:
                c_ = c.split()
                if c_[6] == '<=':
                    cases.append((float(c_[2]), float(c_[5]), float(c_[8]), 'lower'))
                elif c_[6] == '>=':
                    cases.append((float(c_[2]), float(c_[5]), float(c_[8]), 'upper'))
                else:
                    print(c_[6])
                    raise NotImplementedError
            return cases

        old_rv = rv
        rv = []

        for rv_tuple in old_rv:
            if len(conjuncts) > 10:
                pbar = tqdm.tqdm(conjuncts)
            else:
                pbar = conjuncts

            for c in pbar:
                rv_tuple_copy = deepcopy(rv_tuple)
                rv.append(rv_tuple_copy)

                c_tokens = [s for s in c.split(" ") if len(s) > 0]

                count = len(c_tokens) // 3

                for i in range(count):
                    op, first, second = c_tokens[3 * i:3 * (i + 1)]

                    update_rv_tuple(rv_tuple_copy, op, first, second, num_inputs, num_outputs)

    merged_rv = {}

    for rv_tuple in rv:
        boxdict = rv_tuple[0]
        matrhs = (rv_tuple[1], rv_tuple[2])

        key = str(boxdict)

        if key in merged_rv:
            merged_rv[key][1].append(matrhs)
        else:
            merged_rv[key] = (boxdict, [matrhs])

    final_rv = []

    for rv_tuple in merged_rv.values():
        box_dict = rv_tuple[0]

        box = []

        for d in range(num_inputs):
            r = box_dict[d]

            assert r[0] != -np.inf and r[1] != np.inf, f"input X_{d} was unbounded: {r}"
            box.append(r)

        spec_list = []

        for matrhs in rv_tuple[1]:
            mat = np.array(matrhs[0], dtype=float)
            rhs = np.array(matrhs[1], dtype=float)
            spec_list.append((mat, rhs))

        final_rv.append((box, spec_list))

    return final_rv
