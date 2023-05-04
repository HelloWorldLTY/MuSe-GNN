library(ggplot2)

table = readxl::read_excel("total pathway enrichment final.xlsx")

ggplot(table, aes(x=tissuename,y=`pathway term`)) +
  geom_point(aes(color=lnfdr,size=`rich factor`)) +
  scale_color_gradientn(colours = rainbow(5)) +
  labs(
    x='Tissue', y=NULL,
    color='-ln(FDR)',size='Factor'
  ) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = .5, vjust = .6),
    axis.title = element_text(face='bold'),
    axis.text = element_text(face='bold')
  )
