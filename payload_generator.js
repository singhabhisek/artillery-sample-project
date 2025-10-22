// payload_generator.js

const { faker } = require('@faker-js/faker');

// Function to generate a single dummy user record
function generateUserRecord() {
    return {
        name: faker.person.fullName(),
        address: faker.location.streetAddress(true) + ', ' + faker.location.city() + ', ' + faker.location.country(),
        email: faker.internet.email(),
        job: faker.person.jobTitle(),
        company: faker.company.name(),
        phone: faker.phone.number(),
        ssn: faker.helpers.replaceSymbols('###-##-####'),
        blood_group: faker.helpers.arrayElement(['A+', 'B+', 'AB+', 'O+', 'A-', 'B-', 'AB-', 'O-']),
        date_of_birth: faker.date.birthdate().toISOString().split('T')[0],
        transaction_id: faker.string.uuid()
    };
}

/**
 * Custom function to generate a large POST body payload.
 * It creates an array of 10 records to ensure the payload size is ~3KB.
 * This function will be called before the POST request in the YAML file.
 */
module.exports = {
    setPostBody: (requestParams, context, ee, next) => {
        const records = [];
        // Generate 10 records to consistently hit the ~3KB payload size
        for (let i = 0; i < 10; i++) {
            records.push(generateUserRecord());
        }
        
        // Set the generated array as the JSON body for the next request
        context.vars.postBody = records;

        // Must call next() when done
        return next();
    }
};