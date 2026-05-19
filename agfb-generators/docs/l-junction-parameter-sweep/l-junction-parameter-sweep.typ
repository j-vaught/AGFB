#import "@preview/cetz:0.4.2": canvas, draw

#set document(title: "L Junction Parameter Sweep", author: "J.C. Vaught")
#set page(width: 11in, height: 4.1in, margin: 0.32in)
#set text(font: "New Computer Modern", size: 7.4pt)
#set par(leading: 0.35em)

#let garnet = rgb("#73000A")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")

#let data = json("l_junction_parameter_sweep.json")

#let raster(runs, palette) = canvas(length: 1cm, {
  let size = data.size
  let cell = data.cell_size
  for run in runs {
    let x = run.at(0)
    let y = run.at(1)
    let width = run.at(2)
    let level = run.at(3)
    let y0 = (size - y - 1) * cell
    draw.rect(
      (x * cell, y0),
      ((x + width) * cell, y0 + cell),
      fill: rgb(palette.at(level)),
      stroke: none,
    )
  }
  draw.rect((0, 0), (size * cell, size * cell), stroke: black30 + 0.18pt, fill: none)
})

#let legend-swatch(color, label) = [
  #box(width: 0.22in, height: 0.055in, fill: color)
  #h(0.08em)
  #text(fill: black70, size: 6.8pt)[#label]
]

#let panel(item) = block(width: 100%, breakable: false)[
  #text(weight: "bold", fill: black90, size: 7.3pt)[#item.title]
  #v(0.04em)
  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    [#raster(item.intensity, data.palettes.intensity)],
    [#raster(item.gradient, data.palettes.gradient)],
  )
  #v(0.04em)
  #text(fill: black70, size: 6.15pt)[#item.caption]
]

#align(center)[
  #text(size: 14pt, weight: "bold", fill: black90)[L Junction Parameter Sweep]
  #h(1.4em)
  #legend-swatch(garnet, [intensity])
  #h(0.7em)
  #legend-swatch(honeycomb, [gradient magnitude])
]

#v(0.22em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 6pt,
  row-gutter: 7pt,
  ..data.panels.map(panel),
)
