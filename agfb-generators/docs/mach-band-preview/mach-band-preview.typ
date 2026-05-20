#import "@preview/cetz:0.4.2": canvas, draw

#set document(title: "Mach Band Preview", author: "J.C. Vaught")
#set page(width: 10.6in, height: 2.8in, margin: 0.32in)
#set text(font: "New Computer Modern", size: 7.6pt)
#set par(leading: 0.36em)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#let data = json("mach_band_preview.json")

#let to-points(points) = points.map(p => (p.at(0), p.at(1)))

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

#let profile(plot) = canvas(length: 1cm, {
  let plot-width = data.plot_width
  let plot-height = data.plot_height
  draw.rect((0, 0), (plot-width, plot-height), stroke: black30 + 0.2pt, fill: none)
  draw.line((0, plot-height * 0.143), (plot-width, plot-height * 0.143), stroke: black10 + 0.25pt)
  draw.line((0, plot-height * 0.857), (plot-width, plot-height * 0.857), stroke: black10 + 0.25pt)
  draw.line(..to-points(plot.base), stroke: atlantic + 0.65pt)
  draw.line(..to-points(plot.intensity), stroke: garnet + 0.85pt)
})

#let swatch(color, label) = [
  #box(width: 0.18in, height: 0.024in, fill: color)
  #h(0.08em)
  #text(fill: black70, size: 6.7pt)[#label]
]

#let panel(item) = block(width: 100%, breakable: false)[
  #text(weight: "bold", fill: black90, size: 7.7pt)[#item.title]
  #v(0.04em)
  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    [#raster(item.intensity, data.palettes.intensity)],
    [#raster(item.gradient, data.palettes.gradient)],
  )
  #v(0.08em)
  #profile(item.profile)
  #v(0.04em)
  #text(fill: black70, size: 6.05pt)[#item.caption]
]

#align(center)[
  #text(size: 14pt, weight: "bold", fill: black90)[Mach Band Preview]

  #v(0.06em)
  #grid(
    columns: (auto, auto, auto),
    gutter: 9pt,
    swatch(garnet, [Mach band]),
    swatch(atlantic, [base ramp]),
    swatch(honeycomb, [gradient magnitude]),
  )
]

#v(0.20em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  ..data.panels.map(panel),
)
