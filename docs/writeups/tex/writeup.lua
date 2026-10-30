-- pandoc filter for the ml-ladder writeup PDFs.
-- The first blockquote of a writeup is its status block. It moves out of the
-- body into the `status` variable, which the template sets in a status box on
-- the title page, ahead of the table of contents. Horizontal rules become a
-- short accent rule. Inline code may break after _ . / - so long identifiers
-- and paths wrap instead of running into the margin.

function HorizontalRule()
  return pandoc.RawBlock("latex", "\\writeuprule")
end

local escapes = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$",
  ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%", ["_"] = "\\_",
  ["^"] = "\\textasciicircum{}", ["~"] = "\\textasciitilde{}",
}

function Code(el)
  local out = el.text:gsub("[\\{}$&#%%_^~]", escapes)
  out = out:gsub("(\\_)", "%1\\allowbreak{}")
  out = out:gsub("([./-])", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{" .. out .. "}")
end

function Pandoc(doc)
  for i, block in ipairs(doc.blocks) do
    if block.t == "BlockQuote" then
      doc.meta.status = pandoc.MetaBlocks(block.content)
      doc.blocks:remove(i)
      break
    end
  end
  return doc
end

-- Headings keep pandoc's own code handling, which also gives the PDF bookmark
-- a plain-text version; the raw inline above would leave the bookmark empty.
function Header(el)
  return el, false
end

traverse = "topdown"
