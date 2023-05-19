module Data.DatasetGeneration.SemiAutoLabel.Common.Cleaners

open System.Text.RegularExpressions
open  Data.DatasetGeneration.SemiAutoLabel.Common.Patterns


let commonClean (item: string) =
    item
    |> fun t -> Regex.Replace(t, linkPattern, " [LINK] ")
    |> fun t -> Regex.Replace(t, usernamePattern, " [USERNAME] ")
    |> fun t -> t.Replace('\r', ' ')
    |> fun t -> t.Replace('\n', ' ')
    |> fun t -> t.Replace("&amp;", "&")
    |> fun t -> t.Replace("&gt;", ">")
    |> fun t -> t.Replace("&lt;", "<")
    |> fun t -> t.Replace("&eq;", "=")
    |> fun t -> Regex.Replace(t, emojiPattern, " [EMOJI] ")
    |> fun t -> Regex.Replace(t, smileyPattern, " [SMILEY] ")

let aggressivelyClean (item: string) =
    item
    |> fun t ->
        Regex.Matches(t, @"[\u0620-\u06FF]+")
        |> Seq.map (fun m -> m.Value)
        |> String.concat " "
    |> fun t -> Regex.Replace(t, @"[\u064B-\u065F]+", "")
