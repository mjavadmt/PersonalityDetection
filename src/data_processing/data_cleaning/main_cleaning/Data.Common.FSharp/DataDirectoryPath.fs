namespace Data.Common

module DataDirectoryPath =
    [<Literal>]
    let ROOT =
        __SOURCE_DIRECTORY__ + "/../../data"
        
    [<Literal>]
    let RAW =
        ROOT + "/raw"
        
    [<Literal>]
    let INTERIM =
        ROOT + "/interim"
        
    [<Literal>]
    let PROCESSED =
        ROOT + "/processed"
        
    [<Literal>]
    let EXTERNAL =
        ROOT + "/external"