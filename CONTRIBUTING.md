# Contribute

Do you like this project? 
If so, you can contribute to it in various ways, 
and you don't need to be a developer!

Download the latest GitHub version, 
then install the dev requirements in ``requirements-dev.txt``.

In a nutshell

```bash
git clone https://github.com/HugoFara/pylinkage.git
cd pylinkage
pip install -r requirements-dev.txt
```

You will need to have your own fork for this project if you want to submit pull requests.

## Testing

We use unittest. You can use it in two ways:

* Just run ``python -m unittest discover`` from the main folder.
* For users of PyCharm, use the "All Tests" configuration.

## Release

This section is mainly intended for maintainers.
Fell free to use the tools described here, but they are not necessary in any way.

1. To publish a new version, use ``bump2version``. For instance ``bump2version minor``.
2. Update CHANGELOG.md with release date and edit subsection titles.
3. Regenerate the documentation (uses Sphinx).
   * By hand with ``sphinx-build -b html sphinx/ docs/``.
   * We also provide a configuration file for users of PyCharm that does the same.
   * Clean everything with ``make clean``.
4. Commit and add a tag (e. g. v0.4.0).
5. Publish a new [GitHub release](https://github.com/HugoFara/pylinkage/releases).


## Caveats

Pylinkage is a small project to build 2D linkages in a simple way.
It is not intended for complex simulations.
If you want to do much more, my best advice is to start your own project.
If you simply want to have fun developing new features, you are welcome here!

## Contributions for everyone

Don't forget to drop a star, fork it or share it on social media.
This is a community project, and the bigger the community, the more it will thrive!

