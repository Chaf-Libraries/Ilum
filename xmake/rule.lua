rule("empty")
    on_load(function (target)
        if is_mode("debug") then
            target:add("defines", "DEBUG")
        end
    end)

    on_build(function (target, sourcefile)
    end)