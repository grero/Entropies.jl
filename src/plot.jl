import Winston
import Winston.plot
import Colors

function plot(PP::ConditionalProbability;kvs...)
	if length(PP.xybins) >= 2
		ta = Winston.Table(length(PP.xybins[2])+1, length(PP.xybins[1]))
	else
		ta = Winston.Table(1,length(PP.xybins[1]))
	end
	ta.attr[:align_interiors] = true
	ta.attr[:cellspacing] = 2.0
	plot!(ta, PP;kvs...)
	ta
end

function plot!(ta::Winston.Table, PP::ConditionalProbability;kvs...)
	nbins,nsteps = size(PP.pxy)
	nrows,ncols = ta.rows, ta.cols
	x = PP.ybins
	row_offset = 0
        barwidth = Winston.config_value("FramedBar", "barwidth")
        offset = Winston.config_value("FramedBar", "offset")
        bcolor = convert(Colors.LCHuv, Colors.RGB(0.5,1.0,0.5))
	if length(PP.xybins) >= 2
		nrows -= 1
		row_offset = 1
		#reserve first row for marginal over x2
		pxy = zeros(size(PP.pxy,1), length(PP.xybins[1]))
		py = zeros(length(PP.xybins[1]))

		for i in 1:size(PP.pxy,2)
			i1,i2 = ind2sub((length(PP.xybins[1]), length(PP.xybins[2])),i)
			pxy[:,i1] += PP.pxy[:,i]*PP.py[i]
			py[i1] += PP.py[i]
		end

		for i in 1:size(pxy,2)
			pxy[:,i] = pxy[:,i]./ py[i]
		end
		r = 1
		for c in 1:size(pxy,2)
                        if !isdefined(ta.content, r, c)
                            ta[r,c] = Winston.FramedPlot()
                            _color = bcolor
                            barwidth = Winston.config_value("FramedBar", "barwidth")
                            offset = Winston.config_value("FramedBar", "offset")
                        else
                            offset = 0.2
                            barwidth = 0.4
                            ta[r,c].content1.components[1].attr[:offset] = -offset
                            ta[r,c].content1.components[1].attr[:barwidth] = barwidth
                            _color = Colors.LCHuv(bcolor.l, bcolor.c, 360-bcolor.h)
                        end
			if any(isnan(pxy[:,c]))
				Winston.bar(ta[r,c], x, ones(x);color=Colors.RGB(1.0, 1.0, 1.0),draw_baseline=false)
			else
				Winston.bar(ta[r,c], x, pxy[:,c];color=_color,draw_baseline=false)
			end
                        ta[r,c].content1.components[end].attr[:offset] = offset
                        ta[r,c].content1.components[end].attr[:barwidth] = barwidth
			Winston.setattr(ta[r,c], "title", string(PP.xybins[1][c]))
			Winston.setattr(ta[r,c].x2, "draw_axis", false)
			Winston.setattr(ta[r,c].y2, "draw_axis", false)
			Winston.setattr(ta[r,c].x1, "draw_ticklabels", false)
			Winston.setattr(ta[r,c].y1, "draw_ticklabels", false)
			Winston.setattr(ta[r,c].x1, "tickdir", 1)
			Winston.setattr(ta[r,c].y1, "tickdir", 1)
                        Winston.setattr(ta[r,c], "yrange", (0,1))
		end
	end
        bcolor = convert(Colors.LCHuv, Colors.RGB(0.5,0.5,1.0))
	for i in 1:nsteps
		c,r = ind2sub((ncols,nrows), i) #plot sh_dst along the columns and src along the rows
		r += row_offset
                if !isdefined(ta.content, r, c)
                    ta[r,c] = Winston.FramedPlot()
                    _color = bcolor
                else
                    _color = Colors.LCHuv(bcolor.l, bcolor.c, 360-bcolor.h)
                    ta[r,c].content1.components[1].attr[:offset] = -offset
                    ta[r,c].content1.components[1].attr[:barwidth] = barwidth
                end
		if any(isnan(PP.pxy[:,i]))
			Winston.bar(ta[r,c], x, ones(x);color=Colors.RGB(1.0, 1.0, 1.0),draw_baseline=false)
		else
			Winston.bar(ta[r,c], x, PP.pxy[:,i];color=_color,draw_baseline=false)
		end
                ta[r,c].content1.components[end].attr[:offset] = offset
                ta[r,c].content1.components[end].attr[:barwidth] = barwidth
		Winston.setattr(ta[r,c].x2, "draw_axis", false)
		Winston.setattr(ta[r,c].y2, "draw_axis", false)
		Winston.setattr(ta[r,c].x1, "draw_ticklabels", false)
		Winston.setattr(ta[r,c].y1, "draw_ticklabels", false)
		Winston.setattr(ta[r,c].x1, "tickdir", 1)
		Winston.setattr(ta[r,c].y1, "tickdir", 1)
                Winston.setattr(ta[r,c], "yrange", (0,1))
		if r == 1
			Winston.setattr(ta[r,c], "title", string(PP.xybins[1][c]))
		end
		if c == ta.cols
			Winston.setattr(ta[r,c].y2, "draw_axis", true)
			Winston.setattr(ta[r,c].y2, "draw_spine", false)
			Winston.setattr(ta[r,c].y2, "draw_subticks", false)
			if r > 1
				Winston.setattr(ta[r,c].y2,"label", string(PP.xybins[2][r-row_offset]))
			end
		end
		if r == ta.rows && c == 1
			#Winston.setattr(ta[r,c].x1, "ticks", 1.0*[x[1], x[end]])
			#Winston.setattr(ta[r,c].x1, "subticks", 1.0*x)
			Winston.setattr(ta[r,c].x1, "draw_ticklabels", true)
			Winston.setattr(ta[r,c], "ylabel", "P(x)")
		end
	end
end
