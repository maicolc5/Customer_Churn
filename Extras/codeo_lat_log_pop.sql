WITH RankedData AS (
    SELECT
        c.CustomerID,
        a.PostalCode,
        a.City,
        pc.Name AS county_name,
        gc.Latitude,
        gc.Longitude,
        b.populatio AS population,
        ROW_NUMBER() OVER (
            PARTITION BY c.CustomerID
            ORDER BY b.populatio DESC, gc.Latitude ASC, gc.Longitude ASC
        ) AS rn
    FROM
        Sales.Customer c
    LEFT JOIN
        Person.BusinessEntityAddress bea1 ON c.PersonID = bea1.BusinessEntityID
    LEFT JOIN
        Person.BusinessEntityAddress bea2 ON c.StoreID = bea2.BusinessEntityID
    INNER JOIN
        Person.Address a ON a.AddressID = COALESCE(bea1.AddressID, bea2.AddressID)
    LEFT JOIN 
        Person.StateProvince PS ON PS.StateProvinceID = a.StateProvinceID
    LEFT JOIN 
        Person.CountryRegion pc ON pc.CountryRegionCode = ps.CountryRegionCode
    LEFT JOIN GeoData.dbo.geo_customers gc
        ON gc.city COLLATE DATABASE_DEFAULT = a.City COLLATE DATABASE_DEFAULT 
       AND pc.Name COLLATE DATABASE_DEFAULT = gc.CountryRegionName COLLATE DATABASE_DEFAULT
    LEFT JOIN GeoData.dbo.Tmp_WorldCities b
        ON gc.city COLLATE DATABASE_DEFAULT = b.city COLLATE DATABASE_DEFAULT 
       AND gc.CountryRegionName COLLATE DATABASE_DEFAULT = b.country COLLATE DATABASE_DEFAULT
    AND FLOOR(gc.Latitude) = FLOOR(b.lat)
       AND FLOOR(gc.Longitude) = FLOOR(b.lng)
)
SELECT
    CustomerID,
    PostalCode,
    City,
    county_name,
    Latitude,
    Longitude,
    population
INTO GeoData.dbo.CustomerPopulation
FROM RankedData
WHERE rn = 1;

--select distinct City, county_name 
--    from CustomerPopulation
--    where population is NULL

