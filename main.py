import sys
from pathlib import Path
try:
    from openvino.runtime import Core
except ImportError:
    print("Put openvino python API in PYTHONPATH please:")
    sys.exit(1)

def collect_all_irs(directory: Path, ext: str = "xml"):
    return [str(model) for model in directory.rglob("*." + ext) if model.with_suffix(".bin").exists()]

def per_model_collect(core: Core, path: str):
    result = 0
    try:
        model = core.read_model(path)
        for op in model.get_ops():
            if op.type_info.name == 'Transpose':
                result += 1
        return result
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: {} {} directory/with/irs".format(sys.executable, __file__)
    directory = Path(sys.argv[1])
    assert directory.is_dir(), "Not a directory: " + str(directory)
    models = collect_all_irs(directory)
    num_models = len(models)

    core = Core()
    collected_data = dict()
    for i, model in enumerate(sorted(models)):
        collected_data[model] = per_model_collect(core, model)
        print("{:6}% {}".format(round((i + 1) / num_models, 4) * 100, model))

    with open("report.csv", "w+") as f:
        f.write("\n".join(["{},{}".join([model, str(num)]) for model, num in collected_data.items()]))
