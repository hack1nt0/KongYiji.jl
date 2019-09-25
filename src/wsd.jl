
function trim!(m::HMM, trimmeddir::String) {
        @assert isdir(trimmeddir)
        tags = String[]
        wmp = Dict{String, Int}()
        pmp = KongYiji.vec(m.tags)
        p2v = Dict{String, Dict{Int, Tv}}()
        INF2 = Dict{String, Tv}()

        cd(() -> begin
                for tag in readdir(".")
                        ip = get(pmp, tag, 0)
                        if ip == 0; error("Invalid pos-tag in trimmed dir"); end
                        if !isfile(tag); continue; end

                        push!(tags, tag)

                        for line in eachline(tag)
                                word, prob = split(line)
                                word, prob = String(word), parse(Tv, prob)
                                if word == "missing"; INF2[tag] = prob; end
                                iw = getid!(wmp, word)
                                p2v[tag][iw] = prob
                        end
                end

                ips = [pmp[tag] for tag in tags]
                hpr = m.hpr[ips]
                h2h = m.h2h[ips, ips]

                np = length(tags)
                pmp = KongYiji.dict(tags)

                h2v = [Dict{Int, Tv}() for _ in 1:np]
                for (tag, vs) in p2v; h2v[pmp[tag]] = vs; end

                INF = fill(Tv(0), np)
                for (tag, prob) in INF2; INF[pmp[tag]] = prob; end

                normalize!.((hpr, h2h))
                normalize!(h2v, INF)
                log!.((hpr, h2h, h2v, INF))

                m.dict = AhoCorasickAutomaton(wmp)
                m.words = KongYiji.vec(wmp)
                m.usr_words = -1
                m.tags = tags
                m.hpr = hpr
                m.h2h = h2h
                m.h2v = h2v
                m.INF = INF
        end, trimmeddir)
end

function printtrimmed(m::HMM; os=KongYiji.dir("trimmed"))
        np = length(m.tags)
        for ip = 1:np
        	tag = m.tags[ip]
        	open(joinpath(os, tag), "w") do io
        		println(io, "missing", ' ', exp(m.INF[ip]))
   			vs = collect(m.h2v[ip])
                  	sort!(vs; by=last, rev=true)
                  	for (v, p) in vs
                  		v, p = m.words[v], exp(p)
                  		println(io, v, ' ', p)
          		end
		end
	end
end

