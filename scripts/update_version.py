import fire
import toml


def update_version(version):
    file_path = "pyproject.toml"
    with open(file_path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    version = str(version).replace("v", "")
    data["project"]["version"] = version

    with open(file_path, "w", encoding="utf-8") as f:
        toml.dump(data, f)


if __name__ == "__main__":
    fire.Fire(update_version)
