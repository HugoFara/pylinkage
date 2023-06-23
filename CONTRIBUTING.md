# Contribute

Do you like this project? If so, you can contribute to it in various ways, and you don't need to be a developer!

Download the latest GitHub version, then install the dev requirements in ``requirements-dev.txt``.

In a nutshell

```bash
git clone https://github.com/HugoFara/pylinkage.git
cd pylinkage
pip install -r requirements-dev.txt
```

You will need to have your own fork for this project if you want to submit pull requests.

## Testing

We use unittest. Just run ``python3 -m unittest discover .`` from the main folder.

## Release

This section is mainly intended for maintainers.
Fell free to use the tools described here, but they are not necessary in any way.

* To publish a new version, use ``bump2version``. For instance ``bump2version minor``.
* Update CHANGELOG.md with release date and edit subsection titles.
* Regenerate the documentation with ``sphinx-build -b html source docs`` (uses Sphinx). Clean everything with ``make clean``.
* Commit and add a tag (e. g. v0.4.0).
* Publish a new [GitHub release](https://github.com/HugoFara/pylinkage/releases).

## Contributions for everyone

Don't forget to drop a star, fork it or share it on social media.
This is a community project, and the bigger the community, the more it will thrive!
