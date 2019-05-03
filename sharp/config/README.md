Allow the user to create custom `config.py` files, to specify different run
configurations.

- `spec.py` specifies what this config file should look like, and provides
sensible defaults for most all options.

- `load.py` loads this `config.py` file at runtime, so that other modules in
the sharp package can use the loaded config settings.

Because other packages in `sharp` import their settings from this `config`
package, the modules in this package may not import from other packages in
`sharp`, to avoid circular imports.
