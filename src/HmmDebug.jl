

function (hmm::HMM)(input::String, dlm::String)
        standard = split(input, dlm)
        x = join(standard)
        return hmm(x, standard)
end

function (hmm::HMM)(x::String, standard::Vector{<:AbstractString})
        chrs = codeunits(x)
        vtxs = collect(eachmatch(hmm.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length(vtxs), length(chrs), length(hmm.hpr)
        dp = fill(-Inf, (nc + 1, np))
        pre = fill((1, 0), (nc + 1, np))
        for i = 1:nc + 1, j in 1:np dp[i,j] = -Inf end
        pv = 1
        dp[1, :] = hmm.hpr
        pre_i = 1
        for i in 1:nc + 1
                if dp[i,1] != -Inf pre_i = i end
                while pv <= nv && vtxs[pv].s < i pv = pv + 1 end
                if !(pv <= nv && vtxs[pv].s == i || i == nc + 1) continue end
                if dp[i, 1] == -Inf
                        #@show i, pre_i
                        for pi = 1:np, pj = 1:np
                                maybe = dp[pre_i, pj] + hmm.INF[pj] + hmm.h2h[pj,pi]
                                if maybe > dp[i, pi] dp[i, pi] = maybe; pre[i, pi] = (pre_i, pj) end
                        end
                end
                while pv <= nv && vtxs[pv].s == i
                        vtx = vtxs[pv]
                        j = i + length(vtx)
                        for pi = 1:np
                                for pj = 1:np
                                        maybe = dp[i, pi] + hmm.h2h[pi,pj] + get(hmm.h2v[pi], vtx.i, hmm.INF[pi])
                                        if maybe > dp[j,pj] dp[j,pj] = maybe; pre[j,pj] = (i, pi) end
                                end
                        end
                        pv = pv + 1
                end
        end
        v = (nc + 1, argmax(dp[nc + 1,:]))
        output = fill("", 0)
        while v[1] != 1
                pv = pre[v[1],v[2]]
                push!(output, x[pv[1]:prevind(x, v[1])])
                v = pv
        end
        reverse!(output)
        
        #print debug infos
        println("Standard : " * join(standard, "  "))
        println("Output   : " * join(output, "  "))

        v = (nc + 1, argmax(dp[nc + 1,:]))
        info = Matrix{Any}(undef, (length(x), 5))
        nr = 1
        while v[1] != 1
                pv = pre[v[1],v[2]]
                word, postag, source, prob_h2v, prob_add = "", "", "", 0., 0.

                s, t = pv[1], v[1]
                word = x[s:prevind(x, t)]
                postag_id = pv[2]
                postag = hmm.tags[postag_id]
                word_id = 0
                begin
                        vtx_id = searchsortedfirst(vtxs, ACMatch(s, t, -1))
                        if vtx_id < nv + 1 && vtxs[vtx_id].s == s && vtxs[vtx_id].t == t
                                word_id = vtxs[vtx_id].i
                        end
                        if word_id == 0 source = "algorithm"
                        else source = ifelse(word_id > length(hmm.words) - hmm.user_words, "usr.dict", "CTB")
                        end
                end
                prob_h2v = word_id == 0 ? hmm.INF[postag_id] : hmm.h2v[postag_id][word_id]
                prob_add = dp[v[1],v[2]] - dp[pv[1],pv[2]]
                prob_h2v, prob_add = map(x->trunc(exp(x); digits=6), [prob_h2v, prob_add])
                info[nr,:] = [word, postag, source, prob_h2v, prob_add]
                nr += 1
                v = pv
        end
        nr -= 1
        println(UselessTable(reverse(info[1:nr,:]; dims=1); cnames=["word", "pos.tag", "source", "prob.h2v", "Prob.Add."],
                                         heads=["KongYiji(1) Debug Table",], 
                                         foots=["neg.log.likelihood = $(-maximum(dp[nc + 1,:]))"]
                            )
        )
        match_mat = Matrix{Any}(undef, (nv, 3))
        match_mat[:,1] = [(v.s,v.t) for v in vtxs]
        match_mat[:,2] = [x[v] for v in vtxs]
        match_mat[:,3] = [(v.i > length(hmm.words) - hmm.user_words ? "user.dict" : "CTB") for v in vtxs]
        println(UselessTable(match_mat; cnames=["UInt8.range", "word", "source"], heads=["AhoCorasickAutomaton Matched Words"]))

        println(HmmScoreTable(standard, output))
end

function h2vtable(hmm::HMM, postag::String)
        ih = findfirst(isequal(postag), hmm.tags)
        if isnothing(ih) error("$(postag) not found in ctb") end
        vs = collect(hmm.h2v[ih])
        sort!(vs; by=last, rev=true)
        nv = length(vs)
        mat = Matrix{Any}(undef, (nv, 3))
        word_ids, probs = map(first, vs), exp.(map(last, vs))
        mat[:,1] = hmm.words[word_ids]
        mat[:,2] = map(p -> trunc(p, digits=6), probs)
        mat[:,3] = map(i -> (i > length(hmm.words) - hmm.user_words ? "user.dict" : "CTB"), word_ids)
        return UselessTable(mat; cnames=["word", "prob.", "source"], foots=["$(postag) has $(length(vs)) words", "Unknown words porb. = $(1. - sum(probs))"])
end

function v2htable(hmm::HMM, word::String)
        word_id = findfirst(isequal(word), hmm.words)
        if isnothing(word_id) error("$(word) not found in hmm") end
        np = length(hmm.tags)
        hs = [(hmm.tags[first(p)], haskey(last(p), word_id) ? exp(last(p)[word_id]) : 0.) for p in enumerate(hmm.h2v)]
        sort!(hs; by=last, rev=true)
        mat = Matrix{Any}(undef, (np, 1))
        mat[:,1] = map(d -> trunc(d, digits=6), map(last, hs))
        return UselessTable(mat; cnames=["Prob."], rnames=map(first, hs), topleft="From\\To")
end

function h2htable(hmm::HMM)
        np = length(hmm.tags)
        mat = map(d->Int(round(exp(d)*100)), hmm.h2h)
        return UselessTable(mat; cnames=hmm.tags, rnames=hmm.tags, topleft="From\\To(%)")
end

function hprtable(hmm::HMM)
        np = length(hmm.tags)
        mat = Matrix{Any}(undef, (np, 1))
        mat[:,1] = map(d->trunc(exp(d), digits=6), hmm.hpr)
        return UselessTable(mat; cnames=["prob."], rnames=hmm.tags)
end

function postable(hmm::HMM)
        tsv = readdlm(joinpath(pathof(KongYiji), "..", "..", "data", "postable.tsv"), '\t', String)
        return UselessTable(tsv[2:end,:]; cnames=tsv[1,:], heads=["CTB postable"])
end









