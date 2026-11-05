# Albert Sans plot fonts

Paper plots use Albert Sans for labels, ticks, legends and math letters. Fonts are bundled so standalone rendering does not require a system installation. Missing mathematical symbols fall back to STIX; monospace text uses DejaVu Sans Mono.

Source: [Google Fonts Albert Sans](https://github.com/google/fonts/tree/fffdadf0f0c9cc1ec8b407063424a8bfbee05611/ofl/albertsans). See `SOURCE.json` for the pinned revision and source SHA-256 hashes, and `OFL.txt` for the SIL Open Font License.

The source variable fonts are retained. Static regular, semibold, bold and matching italic instances were generated with fontTools 4.60.2 at weights 400, 600 and 700. Their family/style names are set explicitly for Matplotlib selection and PDF embedding.

For a new standalone renderer under `docs`, add the docs directory to `sys.path` and import `paper_plot_style` before creating figures, following the existing renderers. Keep plot-specific sizes and layouts in the renderer. Use a writable `MPLCONFIGDIR` and `XDG_CACHE_HOME`, such as paths under `/tmp`.
