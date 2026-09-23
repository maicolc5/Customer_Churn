

with Poblacion as (select distinct a.*, max(populatio) as population

	from
 GeoData.dbo.geo_customers a
		left join GeoData.dbo.Tmp_WorldCities b on a.city = b.city and  a.CountryRegionName = b.country
		 and FLOOR(a.Latitude) = FLOOR(b.lat) AND FLOOR(a.Longitude) = FLOOR(b.lng)
where populatio is NULL
group by PostalCode, a.City, CountryRegionName, Latitude, Longitude)

select distinct City, CountryRegionName
from Poblacion








