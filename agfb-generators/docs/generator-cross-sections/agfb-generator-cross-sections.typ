#import "@preview/cetz:0.4.2": canvas, draw

#set document(title: "AGFB Generator Cross Sections", author: "J.C. Vaught")
#set page(paper: "us-letter", margin: (x: 0.55in, y: 0.58in))
#set text(font: "New Computer Modern", size: 9.2pt)
#set par(justify: true, leading: 0.55em)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#let data = json("cross_sections.json")

#let to-points(points) = points.map(p => (p.at(0), p.at(1)))

#let legend-line(color, label) = box[
  #canvas(length: 1cm, {
    draw.line((0, 0), (0.55, 0), stroke: color + 0.85pt)
  })
  #h(0.15em)
  #text(size: 8.2pt, fill: black90)[#label]
]

#let cross-section-panel(plot) = block(width: 100%, breakable: false)[
  #text(size: 8.5pt, weight: "bold", fill: black90)[#plot.title]
  #v(0.05em)
  #canvas(length: 1cm, {
    let plot-width = data.plot_width
    let plot-height = data.plot_height
    let mid-y = plot-height / 2

    draw.line((0, 0), (plot-width, 0), stroke: black90 + 0.4pt)
    draw.line((0, 0), (0, plot-height), stroke: black90 + 0.4pt)
    draw.line((0, mid-y), (plot-width, mid-y), stroke: black10 + 0.35pt)
    draw.line((0, plot-height), (plot-width, plot-height), stroke: black30 + 0.25pt)
    draw.line(..to-points(plot.intensity), stroke: garnet + 0.9pt)
    draw.line(..to-points(plot.gradient), stroke: atlantic + 0.7pt)
  })
  #v(0.08em)
  #text(size: 6.9pt, fill: black70)[#plot.caption]
]

#align(center)[
  #text(size: 16pt, weight: "bold")[AGFB Generator Cross Sections]

  #v(0.22em)
  #text(size: 10pt)[J.C. Vaught]

  #text(size: 9pt)[May 19, 2026]
]

#v(0.55em)

This document plots one horizontal image cross section for each public frame generator. The garnet curve is the sampled intensity. The Atlantic curve is the analytic gradient magnitude. The two curves are normalized separately inside each panel, so the plots are for visual profile comparison while the captions report the original intensity range and maximum gradient magnitude.

#v(0.35em)

#legend-line(garnet, [Intensity])
#h(1.2em)
#legend-line(atlantic, [g magnitude])

#v(0.45em)

#for group in data.groups [
  = #group.family

  #grid(
    columns: (1fr, 1fr),
    gutter: 7pt,
    row-gutter: 8pt,
    ..group.plots.map(cross-section-panel),
  )
]
